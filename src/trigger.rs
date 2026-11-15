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
    /// The clip closed. `end` is one past its last frame.
    Close { end: u64, peak_db: f32, ambient_db: f32 },
}

#[derive(Debug, Clone, Copy)]
enum State {
    Idle { above: u32 },
    Open { start: u64, below: u32, peak_db: f32, ambient_db: f32 },
    Cooldown { until: u64 },
}

pub struct Trigger {
    open_over_db: f32,
    close_over_db: f32,
    min_frames: u32,
    hang_frames: u32,
    max_frames: u64,
    cooldown_frames: u64,
    floor_db: f32,
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
            max_frames: 0,
            cooldown_frames: 0,
            floor_db: 0.0,
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

    pub fn is_open(&self) -> bool {
        matches!(self.state, State::Open { .. })
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
            State::Idle { above } if warm && level_db >= open_db => {
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
                if below >= self.hang_frames || n + 1 - start >= self.max_frames {
                    let close = Event::Close { end: n + 1, peak_db, ambient_db };
                    (State::Cooldown { until: n + 1 + self.cooldown_frames }, Some(close))
                } else {
                    (State::Open { start, below, peak_db, ambient_db }, None)
                }
            }
        };
        self.state = next;
        event
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

    #[test]
    fn one_burst_gives_one_clip() {
        let mut t = trigger();
        assert!(feed(&mut t, -50.0, WARMUP_FRAMES).is_empty());
        let events = [feed(&mut t, -30.0, 10), feed(&mut t, -50.0, 200)].concat();
        let start = WARMUP_FRAMES as u64;
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], Event::Open { start });
        // closes after hang_ms (800 ms = 40 frames) below the close line
        assert_eq!(events[1], Event::Close { end: start + 10 + 40, peak_db: -30.0, ambient_db: -50.0 });
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
        let mut t = trigger();
        feed(&mut t, -50.0, WARMUP_FRAMES);
        let events = feed(&mut t, -20.0, 750 + 100 + 3);
        let start = WARMUP_FRAMES as u64;
        assert_eq!(events[0], Event::Open { start });
        assert!(matches!(events[1], Event::Close { end, .. } if end == start + 750));
        // the next clip needs the 2 s cooldown and then min_ms (3 frames) above the line
        assert_eq!(events[2], Event::Open { start: start + 750 + 100 });
        assert_eq!(events.len(), 3);
    }
}
