#include "midi_in.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <mmsystem.h>

namespace mx {
namespace {

std::string narrow(const wchar_t* w) {
    if (!w || !*w) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w, -1, nullptr, 0, nullptr, nullptr);
    std::string s(n > 0 ? n - 1 : 0, 0);
    if (n > 1) WideCharToMultiByte(CP_UTF8, 0, w, -1, s.data(), n, nullptr, nullptr);
    return s;
}

void CALLBACK midi_proc(HMIDIIN, UINT msg, DWORD_PTR instance, DWORD_PTR p1,
                        DWORD_PTR p2) {
    if (msg != MIM_DATA) return;
    auto* self = reinterpret_cast<MidiIn*>(instance);
    if (!self) return;

    const uint8_t status = static_cast<uint8_t>(p1 & 0xF0);
    const uint8_t data1 = static_cast<uint8_t>((p1 >> 8) & 0x7F);
    const uint8_t data2 = static_cast<uint8_t>((p1 >> 16) & 0x7F);

    MidiNote note;
    note.pitch = data1;
    note.velocity = data2;
    note.ms = static_cast<uint32_t>(p2);

    if (status == 0x90 && data2 > 0) {
        note.on = true;
    } else if (status == 0x80 || (status == 0x90 && data2 == 0)) {
        // Note-on with velocity zero is a note-off. Keyboards that use running
        // status send it far more often than a real 0x80, and treating it as an
        // note-on is how every note sticks on forever.
        note.on = false;
    } else {
        return;   // control change, pitch bend, clock -- nothing to show yet
    }
    self->push(note);
}

}  // namespace

MidiIn::~MidiIn() { close(); }

std::vector<std::string> MidiIn::devices() {
    std::vector<std::string> out;
    const UINT count = midiInGetNumDevs();
    for (UINT i = 0; i < count; ++i) {
        MIDIINCAPSW caps{};
        if (midiInGetDevCapsW(i, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
            out.push_back(narrow(caps.szPname));
        else
            out.push_back("device " + std::to_string(i));
    }
    return out;
}

bool MidiIn::open(int index) {
    close();
    error_.clear();

    const UINT count = midiInGetNumDevs();
    if (count == 0) {
        error_ = "no MIDI input device";
        return false;
    }
    const UINT id = index < 0 ? 0u : static_cast<UINT>(index);
    if (id >= count) {
        error_ = "no MIDI device " + std::to_string(index);
        return false;
    }

    HMIDIIN handle = nullptr;
    const MMRESULT r = midiInOpen(&handle, id, reinterpret_cast<DWORD_PTR>(&midi_proc),
                                  reinterpret_cast<DWORD_PTR>(this), CALLBACK_FUNCTION);
    if (r != MMSYSERR_NOERROR) {
        error_ = "could not open the MIDI device (it may be in use)";
        return false;
    }

    MIDIINCAPSW caps{};
    if (midiInGetDevCapsW(id, &caps, sizeof(caps)) == MMSYSERR_NOERROR)
        name_ = narrow(caps.szPname);

    head_.store(0, std::memory_order_relaxed);
    tail_.store(0, std::memory_order_relaxed);
    dropped_.store(0, std::memory_order_relaxed);
    handle_ = handle;
    // Marked open before starting: the driver may deliver a message between
    // midiInStart returning and the next line running.
    open_.store(true, std::memory_order_release);
    midiInStart(handle);
    return true;
}

void MidiIn::close() {
    if (!handle_) return;
    open_.store(false, std::memory_order_release);
    auto handle = static_cast<HMIDIIN>(handle_);
    midiInStop(handle);
    midiInReset(handle);
    midiInClose(handle);   // returns only once the callback is done
    handle_ = nullptr;
    name_.clear();
}

void MidiIn::push(const MidiNote& note) {
    if (!open_.load(std::memory_order_acquire)) return;
    const uint64_t head = head_.load(std::memory_order_relaxed);
    const uint64_t tail = tail_.load(std::memory_order_acquire);
    if (head - tail >= kCapacity) {
        // Five hundred events is about ten seconds of dense playing against a
        // 60 Hz drain. Overflowing means the UI has stopped, and dropping the
        // newest is better than overwriting a note-off and leaving a note on.
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return;
    }
    ring_[static_cast<size_t>(head % kCapacity)] = note;
    head_.store(head + 1, std::memory_order_release);
}

std::vector<MidiNote> MidiIn::drain() {
    std::vector<MidiNote> out;
    const uint64_t head = head_.load(std::memory_order_acquire);
    uint64_t tail = tail_.load(std::memory_order_relaxed);
    out.reserve(static_cast<size_t>(head - tail));
    while (tail < head) {
        out.push_back(ring_[static_cast<size_t>(tail % kCapacity)]);
        ++tail;
    }
    tail_.store(tail, std::memory_order_release);
    return out;
}

}  // namespace mx
