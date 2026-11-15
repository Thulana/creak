//! Ambient-relative sound trigger. Works on 20 ms frame levels only, so it is pure and
//! testable; cutting the audio is the caller's job.

use crate::config;
use std::collections::VecDeque;

pub const SAMPLE_RATE: u32 = 16_000;
pub const FRAME: usize = 320; // 20 ms at 16 kHz
pub const FRAME_MS: u32 = 20;

const AMBIENT_WINDOW_FRAMES: usize = 5 * 60 * 1000 / FRAME_MS as usize;
const AMBIENT_PERCENTILE: f32 = 0.10;
/// No trigger until this much history exists, so the first ambient estimate is meaningful.
const WARMUP_FRAMES: usize = 5 * 1000 / FRAME_MS as usize;
const MIN_DB: f32 = -120.0;
const BIN_DB: f32 = 0.5;
const BINS: usize = (-MIN_DB / BIN_DB) as usize + 1;

pub fn level_db(frame: &[f32]) -> f32 {
    let sum: f32 = frame.iter().map(|s| s * s).sum();
    let rms = (sum / frame.len().max(1) as f32).sqrt();
    (20.0 * rms.max(1e-6).log10()).max(MIN_DB)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Event {
    /// A clip opened. `start` is the first frame of the run that crossed the threshold.
    Open { start: u64 },
    /// The clip closed. `end` is one past its last frame and `sound_end` one past the frame at
    /// which the sound was over; the frames in between are the tail.
    Close { end: u64, sound_end: u64, peak_db: f32, ambient_db: f32 },
}

/// In `Tail` the sound is over but recording goes on until `until`; a new sound before then
/// reopens the clip.
#[derive(Debug, Clone, Copy)]
enum State {
    Idle { above: u32 },
    Open { start: u64, below: u32, peak_db: f32, ambient_db: f32 },
    Tail { start: u64, sound_end: u64, until: u64, above: u32, peak_db: f32, ambient_db: f32 },
    Cooldown { until: u64 },
}

pub struct Trigger {
    open_over_db: f32,
    close_over_db: f32,
    min_frames: u32,
    hang_frames: u32,
    tail_frames: u64,
    min_clip_frames: u64,
    max_frames: u64,
    cooldown_frames: u64,
    floor_db: f32,
    armed: bool,
    history: VecDeque<u16>,
    histogram: Vec<u32>,
    frame: u64,
    state: State,
}

impl Trigger {
    pub fn new(c: &config::Trigger) -> Self {
        let mut trigger = Self {
            open_over_db: 0.0,
            close_over_db: 0.0,
            min_frames: 0,
            hang_frames: 0,
            tail_frames: 0,
            min_clip_frames: 0,
            max_frames: 0,
            cooldown_frames: 0,
            floor_db: 0.0,
            armed: true,
            history: VecDeque::with_capacity(AMBIENT_WINDOW_FRAMES),
            histogram: vec![0; BINS],
            frame: 0,
            state: State::Idle { above: 0 },
        };
        trigger.retune(c);
        trigger
    }

    /// Applies new settings without losing the ambient history or an open clip.
    pub fn retune(&mut self, c: &config::Trigger) {
        let frames = |ms: u32| ms.div_ceil(FRAME_MS).max(1);
        self.open_over_db = 20.0 * c.ratio.log10();
        self.close_over_db = 20.0 * c.close_ratio.log10();
        self.min_frames = frames(c.min_ms);
        self.hang_frames = frames(c.hang_ms);
        self.tail_frames = c.tail_ms.div_ceil(FRAME_MS) as u64;
        self.min_clip_frames = (c.min_clip_s * 1000 / FRAME_MS) as u64;
        self.max_frames = frames(c.max_clip_s * 1000) as u64;
        self.cooldown_frames = (c.cooldown_s * 1000 / FRAME_MS) as u64;
        self.floor_db = c.ambient_floor_db;
    }

    /// The 10th percentile of the trailing 5 minutes, before the floor is applied.
    pub fn measured_ambient_db(&self) -> Option<f32> {
        let target = (self.history.len() as f32 * AMBIENT_PERCENTILE).ceil().max(1.0) as u32;
        let mut seen = 0;
        for (bin, &count) in self.histogram.iter().enumerate() {
            seen += count;
            if seen >= target {
                return Some(MIN_DB + bin as f32 * BIN_DB);
            }
        }
        None
    }

    /// Ambient level in dBFS that the thresholds are relative to: the measured one, never below the floor.
    pub fn ambient_db(&self) -> f32 {
        self.measured_ambient_db().map_or(self.floor_db, |m| m.max(self.floor_db))
    }

    /// (open, close) thresholds in dBFS at the current ambient.
    pub fn thresholds_db(&self) -> (f32, f32) {
        let ambient = self.ambient_db();
        (ambient + self.open_over_db, ambient + self.close_over_db)
    }

    /// True while a clip is being recorded, its tail included.
    pub fn is_recording(&self) -> bool {
        matches!(self.state, State::Open { .. } | State::Tail { .. })
    }

    /// A disarmed trigger keeps measuring the ambient level but opens no clip, and drops the
    /// one it is recording.
    pub fn set_armed(&mut self, armed: bool) {
        if !armed {
            self.state = State::Idle { above: 0 };
        }
        self.armed = armed;
    }

    pub fn push(&mut self, level_db: f32) -> Option<Event> {
        let n = self.frame;
        let (open_db, close_db) = self.thresholds_db();
        let ambient_db = self.ambient_db();
        let warm = self.history.len() >= WARMUP_FRAMES;
        self.record(level_db);
        self.frame += 1;

        let (next, event) = match self.state {
            State::Cooldown { until } if n + 1 >= until => (State::Idle { above: 0 }, None),
            State::Cooldown { .. } => (self.state, None),
            State::Idle { above } if warm && self.armed && level_db >= open_db => {
                let above = above + 1;
                if above >= self.min_frames {
                    let start = n + 1 - above as u64;
                    (State::Open { start, below: 0, peak_db: level_db, ambient_db }, Some(Event::Open { start }))
                } else {
                    (State::Idle { above }, None)
                }
            }
            State::Idle { .. } => (State::Idle { above: 0 }, None),
            State::Open { start, below, peak_db, ambient_db } => {
                let below = if level_db < close_db { below + 1 } else { 0 };
                let peak_db = peak_db.max(level_db);
                let end = n + 1;
                if end - start >= self.max_frames {
                    self.close(end, end, peak_db, ambient_db)
                } else if below >= self.hang_frames {
                    let until = (end + self.tail_frames).max(start + self.min_clip_frames).min(start + self.max_frames);
                    if until <= end {
                        self.close(end, end, peak_db, ambient_db)
                    } else {
                        (State::Tail { start, sound_end: end, until, above: 0, peak_db, ambient_db }, None)
                    }
                } else {
                    (State::Open { start, below, peak_db, ambient_db }, None)
                }
            }
            State::Tail { start, sound_end, until, above, peak_db, ambient_db } => {
                let above = if level_db >= open_db { above + 1 } else { 0 };
                let peak_db = peak_db.max(level_db);
                if above >= self.min_frames {
                    (State::Open { start, below: 0, peak_db, ambient_db }, None)
                } else if n + 1 >= until {
                    self.close(n + 1, sound_end, peak_db, ambient_db)
                } else {
                    (State::Tail { start, sound_end, until, above, peak_db, ambient_db }, None)
                }
            }
        };
        self.state = next;
        event
    }

    fn close(&self, end: u64, sound_end: u64, peak_db: f32, ambient_db: f32) -> (State, Option<Event>) {
        let close = Event::Close { end, sound_end, peak_db, ambient_db };
        (State::Cooldown { until: end + self.cooldown_frames }, Some(close))
    }

    fn record(&mut self, level_db: f32) {
        let bin = (((level_db - MIN_DB) / BIN_DB) as usize).min(BINS - 1) as u16;
        if self.history.len() == AMBIENT_WINDOW_FRAMES {
            let old = self.history.pop_front().unwrap();
            self.histogram[old as usize] -= 1;
        }
        self.history.push_back(bin);
        self.histogram[bin as usize] += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn trigger() -> Trigger {
        Trigger::new(&config::Trigger::default())
    }

    fn feed(t: &mut Trigger, db: f32, frames: usize) -> Vec<Event> {
        (0..frames).filter_map(|_| t.push(db)).collect()
    }

    #[test]
    fn level_of_full_scale_square_is_zero_db() {
        assert!((level_db(&[1.0, -1.0, 1.0, -1.0]) - 0.0).abs() < 1e-4);
        assert_eq!(level_db(&[0.0; 320]), -120.0);
    }

    #[test]
    fn ambient_is_low_percentile_with_floor() {
        let mut t = trigger();
        feed(&mut t, -50.0, 900);
        feed(&mut t, -20.0, 100);
        assert_eq!(t.ambient_db(), -50.0);
        let mut quiet = trigger();
        feed(&mut quiet, -90.0, 500);
        assert_eq!(quiet.ambient_db(), -60.0);
    }

    #[test]
    fn retune_moves_the_line_and_keeps_history() {
        let mut t = trigger();
        feed(&mut t, -50.0, WARMUP_FRAMES);
        t.retune(&config::Trigger { ratio: 10.0, ..Default::default() });
        assert_eq!(t.thresholds_db().0, -30.0);
        assert!(feed(&mut t, -35.0, 10).is_empty());
        assert!(matches!(feed(&mut t, -25.0, 3)[..], [Event::Open { .. }]));
    }

    fn warm(c: config::Trigger) -> Trigger {
        let mut t = Trigger::new(&c);
        assert!(feed(&mut t, -50.0, WARMUP_FRAMES).is_empty());
        t
    }

    #[test]
    fn one_burst_gives_one_clip() {
        let mut t = warm(config::Trigger { tail_ms: 0, min_clip_s: 0, ..Default::default() });
        let events = [feed(&mut t, -30.0, 10), feed(&mut t, -50.0, 200)].concat();
        let start = WARMUP_FRAMES as u64;
        // the sound is over after hang_ms (800 ms = 40 frames) below the close line
        let end = start + 10 + 40;
        let close = Event::Close { end, sound_end: end, peak_db: -30.0, ambient_db: -50.0 };
        assert_eq!(events, vec![Event::Open { start }, close]);
    }

    #[test]
    fn the_tail_keeps_recording_after_the_sound() {
        let mut t = warm(config::Trigger { tail_ms: 1000, min_clip_s: 0, ..Default::default() });
        let events = [feed(&mut t, -30.0, 10), feed(&mut t, -50.0, 200)].concat();
        let start = WARMUP_FRAMES as u64;
        let sound_end = start + 10 + 40;
        let close = Event::Close { end: sound_end + 50, sound_end, peak_db: -30.0, ambient_db: -50.0 };
        assert_eq!(events, vec![Event::Open { start }, close]);
        assert!(!t.is_recording());
    }

    #[test]
    fn a_short_sound_runs_to_the_shortest_clip() {
        // defaults: 2 s tail, but at least 15 s (750 frames) from the onset
        let mut t = warm(config::Trigger::default());
        let events = [feed(&mut t, -30.0, 10), feed(&mut t, -50.0, 800)].concat();
        let start = WARMUP_FRAMES as u64;
        let close = Event::Close { end: start + 750, sound_end: start + 50, peak_db: -30.0, ambient_db: -50.0 };
        assert_eq!(events, vec![Event::Open { start }, close]);
    }

    #[test]
    fn a_sound_in_the_tail_joins_the_clip() {
        let mut t = warm(config::Trigger { tail_ms: 2000, min_clip_s: 0, ..Default::default() });
        // over at +50, then a second burst at +70 while the tail runs to +150
        let events =
            [feed(&mut t, -30.0, 10), feed(&mut t, -50.0, 60), feed(&mut t, -25.0, 10), feed(&mut t, -50.0, 300)]
                .concat();
        let start = WARMUP_FRAMES as u64;
        let sound_end = start + 80 + 40;
        let close = Event::Close { end: sound_end + 100, sound_end, peak_db: -25.0, ambient_db: -50.0 };
        assert_eq!(events, vec![Event::Open { start }, close]);
    }

    #[test]
    fn a_disarmed_trigger_drops_its_clip_and_opens_none() {
        let mut t = warm(config::Trigger::default());
        assert_eq!(feed(&mut t, -30.0, 10).len(), 1);
        t.set_armed(false);
        assert!(!t.is_recording());
        assert!(feed(&mut t, -30.0, 300).is_empty());
        t.set_armed(true);
        assert!(matches!(feed(&mut t, -20.0, 3)[..], [Event::Open { .. }]));
    }

    #[test]
    fn shorter_than_min_ms_does_not_open() {
        let mut t = trigger();
        feed(&mut t, -50.0, WARMUP_FRAMES);
        assert!([feed(&mut t, -30.0, 2), feed(&mut t, -50.0, 100)].concat().is_empty());
    }

    #[test]
    fn no_trigger_during_warmup() {
        let mut t = trigger();
        feed(&mut t, -50.0, 10);
        assert!(feed(&mut t, -20.0, 20).is_empty());
    }

    #[test]
    fn long_noise_is_cut_at_max_and_cooled_down() {
        let mut t = warm(config::Trigger::default());
        // Enough quiet history that a minute of noise does not become the ambient level.
        feed(&mut t, -50.0, 750);
        let events = feed(&mut t, -20.0, 3000 + 100 + 3);
        let start = (WARMUP_FRAMES + 750) as u64;
        assert_eq!(events[0], Event::Open { start });
        // cut at max_clip_s (60 s = 3000 frames) with no tail, because the sound is still going
        assert!(matches!(events[1], Event::Close { end, sound_end, .. } if end == start + 3000 && sound_end == end));
        // the next clip needs the 2 s cooldown and then min_ms (3 frames) above the line
        assert_eq!(events[2], Event::Open { start: start + 3000 + 100 });
        assert_eq!(events.len(), 3);
    }
}
