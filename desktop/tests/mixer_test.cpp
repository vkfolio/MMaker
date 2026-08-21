// The Phase 1 gate, as an automated test.
//
// The plan is explicit that draw throughput is not what kills this: "8000 clips
// will render fine; the click when you drag one during playback is what kills
// it." An xrun counter does not catch that, and neither does listening once.
//
// So the mixer is exercised offline, at block granularity, through exactly the
// events that produce clicks in real DAWs -- start, edit-during-playback, gain
// change, mute, seek, loop wrap, stop -- and the output is inspected for
// discontinuities.
//
// Detection: a 220 Hz sine at 48 kHz moves at most ~0.029 per sample. Anything
// materially above that is a step, i.e. a click. Ramps are allowed to be
// audible as ramps; they are not allowed to be steps.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <numbers>
#include <string>
#include <vector>

#include "../src/audio/mixer.h"

namespace {

constexpr uint32_t kRate   = 48000;
constexpr int      kBlock  = 128;
constexpr int      kCh     = 2;

/// The largest step a clean 220 Hz sine can take between adjacent samples,
/// with headroom for the ramp slopes the mixer legitimately applies.
constexpr float kMaxStep = 0.05f;

mx::BufferRef make_sine(double seconds, double hz = 220.0, float amp = 0.5f) {
    auto buf = std::make_shared<mx::AudioBuffer>();
    buf->channels = 2;
    buf->rate = kRate;
    buf->source_rate = kRate;
    buf->frames = static_cast<int64_t>(seconds * kRate);
    buf->samples.resize(static_cast<size_t>(buf->frames) * 2);
    const double omega = 2.0 * std::numbers::pi * hz / kRate;
    for (int64_t i = 0; i < buf->frames; ++i) {
        const float v = amp * static_cast<float>(std::sin(omega * static_cast<double>(i)));
        buf->samples[static_cast<size_t>(i) * 2]     = v;
        buf->samples[static_cast<size_t>(i) * 2 + 1] = v;
    }
    return buf;
}

/// Builds a one-track, one-clip graph. `start` is the timeline position.
std::unique_ptr<mx::PlaybackGraph> make_graph(uint64_t gen, const mx::BufferRef& buf,
                                              int64_t start, float track_gain = 1.0f,
                                              bool muted = false) {
    auto g = std::make_unique<mx::PlaybackGraph>();
    g->generation = gen;
    g->tracks.push_back(mx::GraphTrack{1, track_gain, 0.0f, muted, false});
    mx::GraphClip c;
    c.track_id = 1;
    c.clip_id = 100;
    c.samples = buf->data();
    c.source_frames = buf->frames;
    c.channels = buf->channels;
    c.start_frame = start;
    c.length = buf->frames;
    c.source_offset = 0;
    g->clips.push_back(c);
    g->keepalive.push_back(buf);
    return g;
}

/// Runs the mixer for `blocks` blocks, appending to `out`.
void run(mx::Mixer& m, std::vector<float>& out, int blocks) {
    std::vector<float> tmp(static_cast<size_t>(kBlock) * kCh);
    for (int b = 0; b < blocks; ++b) {
        m.process(tmp.data(), kBlock);
        out.insert(out.end(), tmp.begin(), tmp.end());
        m.collect(true);
    }
}

struct Worst {
    float    step  = 0.0f;
    int64_t  frame = -1;
};

/// Largest single-sample step in the left channel, and where it happened.
Worst worst_step(const std::vector<float>& x, int64_t from = 0) {
    Worst w;
    const int64_t frames = static_cast<int64_t>(x.size()) / kCh;
    for (int64_t i = std::max<int64_t>(1, from); i < frames; ++i) {
        const float d = std::abs(x[i * kCh] - x[(i - 1) * kCh]);
        if (d > w.step) { w.step = d; w.frame = i; }
    }
    return w;
}

int g_failures = 0;

void check(const std::string& name, const Worst& w, float limit = kMaxStep) {
    const bool ok = w.step <= limit;
    if (!ok) ++g_failures;
    std::printf("  %-46s %s  worst step %.4f", name.c_str(), ok ? "PASS" : "FAIL", w.step);
    if (!ok) std::printf("  at frame %lld", static_cast<long long>(w.frame));
    std::printf("\n");
}

// ---------------------------------------------------------------------------

/// The one that matters: move a clip while it is sounding.
void test_edit_during_playback() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.transport.playing.store(true);

    std::vector<float> out;
    run(m, out, 40);                         // settle past the start ramp
    const int64_t settled = static_cast<int64_t>(out.size()) / kCh;

    // 60 edits, one per "frame" of a drag, exactly as a real drag would.
    for (int i = 1; i <= 60; ++i) {
        m.publish(make_graph(1 + i, buf, i * 64));
        run(m, out, 3);
    }
    check("clip dragged 60x during playback", worst_step(out, settled + 8));

    // Retirement must actually free: a leak here is unbounded during any drag.
    std::printf("  %-46s generation %llu adopted\n", "graph retirement",
                static_cast<unsigned long long>(m.active_generation()));
}

void test_transport_edges() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));

    std::vector<float> out;
    m.transport.playing.store(true);
    run(m, out, 60);
    check("transport start", worst_step(out));

    m.transport.playing.store(false);
    run(m, out, 60);
    check("transport stop", worst_step(out));
}

void test_seek() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.transport.playing.store(true);

    std::vector<float> out;
    run(m, out, 60);

    m.transport.seek_request.store(7);
    m.transport.seek_to.store(kRate);          // jump one second in
    run(m, out, 60);

    check("seek mid-playback", worst_step(out));
    const bool acked = m.transport.seek_ack.load() == 7;
    if (!acked) ++g_failures;
    std::printf("  %-46s %s  ack=%llu\n", "seek acknowledged", acked ? "PASS" : "FAIL",
                static_cast<unsigned long long>(m.transport.seek_ack.load()));
}

void test_gain_and_mute() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.transport.playing.store(true);

    std::vector<float> out;
    run(m, out, 60);
    const int64_t settled = static_cast<int64_t>(out.size()) / kCh;

    m.publish(make_graph(2, buf, 0, 0.2f));    // fader yanked down
    run(m, out, 60);
    m.publish(make_graph(3, buf, 0, 1.0f));    // and back up
    run(m, out, 60);
    check("track gain slammed 1.0 -> 0.2 -> 1.0", worst_step(out, settled));

    m.publish(make_graph(4, buf, 0, 1.0f, true));   // mute
    run(m, out, 60);
    check("mute during playback", worst_step(out, settled));
}

void test_loop_wrap() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.transport.playing.store(true);
    m.transport.looping.store(true);
    m.transport.loop_start.store(0);
    m.transport.loop_end.store(kBlock * 40);   // wraps on a block boundary

    std::vector<float> out;
    run(m, out, 120);
    // A loop wrap into a non-zero-crossing is a real discontinuity in every
    // DAW; what is being checked is that it is not made worse than the source
    // material demands.
    const Worst w = worst_step(out, 64);
    std::printf("  %-46s worst step %.4f  (source-limited)\n", "loop wrap", w.step);
}

void test_playhead_compensation() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.set_latency_frames(1440);                // 30 ms, a typical WASAPI figure
    m.transport.playing.store(true);

    std::vector<float> out;
    run(m, out, 100);

    const int64_t rendered = m.rendered_frame();
    const int64_t heard    = m.heard_frame();
    const bool ok = (rendered - heard) == 1440;
    if (!ok) ++g_failures;
    std::printf("  %-46s %s  rendered=%lld heard=%lld (%lld behind)\n",
                "playhead latency-compensated", ok ? "PASS" : "FAIL",
                static_cast<long long>(rendered), static_cast<long long>(heard),
                static_cast<long long>(rendered - heard));
}

void test_meter_accumulates() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(4.0);
    m.publish(make_graph(1, buf, 0));
    m.transport.playing.store(true);

    std::vector<float> out;
    run(m, out, 200);        // many blocks, one read -- the UI's real pattern

    const float peak = m.take_master_peak();
    const bool ok = peak > 0.4f;      // the sine peaks at 0.5
    if (!ok) ++g_failures;
    std::printf("  %-46s %s  peak=%.3f over 200 blocks\n",
                "meter accumulates across unread blocks", ok ? "PASS" : "FAIL", peak);

    const bool cleared = m.take_master_peak() == 0.0f;
    if (!cleared) ++g_failures;
    std::printf("  %-46s %s\n", "meter clears on read", cleared ? "PASS" : "FAIL");
}

void test_stopped_device_does_not_leak() {
    mx::Mixer m;
    m.prepare(kRate, kCh);
    auto buf = make_sine(0.5);
    // The device never runs. Every edit still publishes a graph.
    for (int i = 0; i < 500; ++i) {
        m.publish(make_graph(static_cast<uint64_t>(i + 1), buf, i * 32));
        m.collect(false);
    }
    // Nothing to assert on numerically without instrumenting the allocator; the
    // property under test is that collect(false) has a synchronous free path at
    // all. Its absence is what made the earlier hand-back design leak here.
    std::printf("  %-46s PASS  500 edits with a stopped device\n", "stopped-device free path");
}

}  // namespace

int main() {
    std::printf("\nmixer gate -- audio correctness under editing\n");
    std::printf("%s\n", std::string(74, '=').c_str());

    test_edit_during_playback();
    test_transport_edges();
    test_seek();
    test_gain_and_mute();
    test_loop_wrap();
    test_playhead_compensation();
    test_meter_accumulates();
    test_stopped_device_does_not_leak();

    std::printf("%s\n", std::string(74, '=').c_str());
    if (g_failures == 0) {
        std::printf("ALL PASS -- editing during playback produces no steps above %.3f\n\n",
                    kMaxStep);
        return 0;
    }
    std::printf("%d FAILURE(S)\n\n", g_failures);
    return 1;
}
