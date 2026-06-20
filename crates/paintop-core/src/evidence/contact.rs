//! The before/after/diff contact-sheet compositor (`plan.md` §15.1
//! `contact-sheet.png`, §18.2).
//!
//! A contact sheet is the one image an agent (or human) glances at to see what a
//! run *did*: the input, the output, and a diff that makes the change legible.
//! This module composites those three panels side by side into a single
//! deterministic RGBA raster and encodes it with the in-crate
//! [`png`](crate::evidence::png) writer, so re-running a plan yields a
//! byte-identical `contact-sheet.png`.
//!
//! ## Panels
//!
//! * **before** — the input raster.
//! * **after** — the output raster.
//! * **diff** — a generated visualization of the per-pixel change between them.
//!
//! When before and after share dimensions the diff is the per-channel absolute
//! difference, amplified by a fixed gain so a tiny localized change is visible
//! (the amplification is deterministic, not a heuristic that drifts). When they
//! differ in size the diff panel is filled with a neutral background — the sheet
//! is still well-formed, it simply cannot show a pixel-aligned diff. The sheet is
//! laid out at the **maximum** panel height, with shorter panels top-aligned over
//! a neutral background, so panels of different sizes still compose into one
//! rectangle.
//!
//! ## Optionality
//!
//! A contact sheet is a *failure-relevant* / requested artifact: if the inputs
//! needed to build it are absent, no sheet is written and the bundle is still
//! valid (`plan.md` §15.1 acceptance — a missing optional artifact is absent, not
//! malformed). [`ContactSheet::compose`] therefore returns the bytes for the
//! caller to write only when it has the panels.

use crate::evidence::png::encode_rgba;

/// The neutral background filling unused panel area (opaque mid-grey).
const BACKGROUND: [u8; 4] = [32, 32, 32, 255];

/// The fixed gain applied to per-channel absolute differences in the diff panel,
/// so small localized changes are visible. Deterministic by construction.
const DIFF_GAIN: u8 = 8;

/// One RGBA8 panel of a contact sheet: a row-major `width × height` raster with
/// 4 bytes per pixel.
///
/// Construction validates that the buffer length matches the dimensions, so a
/// malformed panel is rejected up front rather than producing a corrupt sheet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Panel {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl Panel {
    /// Wrap a raster as a panel, validating `rgba.len() == width * height * 4`.
    ///
    /// Returns `None` for a zero dimension or a buffer whose length does not
    /// match the dimensions.
    #[must_use]
    pub fn new(width: u32, height: u32, rgba: Vec<u8>) -> Option<Self> {
        let expected = (width as usize)
            .checked_mul(height as usize)?
            .checked_mul(4)?;
        if width == 0 || height == 0 || rgba.len() != expected {
            return None;
        }
        Some(Self {
            width,
            height,
            rgba,
        })
    }

    /// The panel width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The panel height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Encode this panel as a deterministic RGBA8 PNG.
    ///
    /// Returns `None` only for a degenerate (zero-area) panel, which
    /// [`Panel::new`] already rejects.
    #[must_use]
    pub fn encode_png(&self) -> Option<Vec<u8>> {
        encode_rgba(self.width, self.height, &self.rgba)
    }

    /// The before/after **diff panel** on its own — the amplified per-pixel
    /// change visualization, without the before/after panels beside it.
    ///
    /// This is the standalone "outside diff" artifact a failure materializes
    /// (`plan.md` §18.2): the same diff that appears in a full contact sheet, but
    /// isolated so it can be written under `diffs/`.
    #[must_use]
    pub fn diff_panel(before: &Self, after: &Self) -> Self {
        Self::diff(before, after)
    }

    /// The RGBA value at pixel `(x, y)`, or the background outside the panel.
    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        if x >= self.width || y >= self.height {
            return BACKGROUND;
        }
        let idx = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        [
            self.rgba[idx],
            self.rgba[idx + 1],
            self.rgba[idx + 2],
            self.rgba[idx + 3],
        ]
    }

    /// Build the diff panel between `before` and `after`.
    ///
    /// When the two share dimensions, each output channel is the amplified
    /// absolute difference of the inputs (alpha forced opaque). When they differ,
    /// the diff is a neutral panel sized to the larger footprint so the sheet
    /// stays rectangular.
    fn diff(before: &Self, after: &Self) -> Self {
        let width = before.width.max(after.width);
        let height = before.height.max(after.height);
        let mut rgba = vec![0u8; (width as usize) * (height as usize) * 4];
        let aligned = before.width == after.width && before.height == after.height;
        for y in 0..height {
            for x in 0..width {
                let px = if aligned {
                    let b = before.pixel(x, y);
                    let a = after.pixel(x, y);
                    [
                        amplified_delta(b[0], a[0]),
                        amplified_delta(b[1], a[1]),
                        amplified_delta(b[2], a[2]),
                        255,
                    ]
                } else {
                    BACKGROUND
                };
                let idx = ((y as usize) * (width as usize) + (x as usize)) * 4;
                rgba[idx..idx + 4].copy_from_slice(&px);
            }
        }
        Self {
            width,
            height,
            rgba,
        }
    }
}

/// The amplified absolute difference of two channel samples, saturating at 255.
const fn amplified_delta(a: u8, b: u8) -> u8 {
    a.abs_diff(b).saturating_mul(DIFF_GAIN)
}

/// A composited before/after/diff contact sheet, ready to encode.
///
/// The three panels are laid out left to right at the maximum panel height;
/// shorter panels are top-aligned over the neutral background. The diff panel is
/// generated from the before/after pair (see [`Panel::diff_panel`]).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContactSheet {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl ContactSheet {
    /// The conventional bundle-relative path of the contact sheet artifact.
    pub const PATH: &'static str = "contact-sheet.png";

    /// Composite a before/after/diff sheet from the `before` and `after` panels.
    ///
    /// The diff panel is generated internally, so the caller supplies only the
    /// two rasters it actually has. The sheet is `before.width + after.width +
    /// diff.width` wide and `max(height)` tall.
    #[must_use]
    pub fn compose(before: &Panel, after: &Panel) -> Self {
        let diff = Panel::diff(before, after);
        Self::lay_out(&[before, after, &diff])
    }

    /// Lay `panels` left to right at the maximum height, top-aligned over the
    /// neutral background.
    fn lay_out(panels: &[&Panel]) -> Self {
        let total_width: u32 = panels.iter().map(|p| p.width).sum();
        let height = panels.iter().map(|p| p.height).max().unwrap_or(0);
        let mut rgba = vec![0u8; (total_width as usize) * (height as usize) * 4];
        let mut x_offset = 0u32;
        for panel in panels {
            for y in 0..height {
                for x in 0..panel.width {
                    let px = panel.pixel(x, y);
                    let dest_x = x_offset + x;
                    let idx = ((y as usize) * (total_width as usize) + (dest_x as usize)) * 4;
                    rgba[idx..idx + 4].copy_from_slice(&px);
                }
            }
            x_offset += panel.width;
        }
        // Background-fill any pixel left untouched by a shorter panel.
        Self::fill_gaps(&mut rgba, total_width, height, panels);
        Self {
            width: total_width,
            height,
            rgba,
        }
    }

    /// Fill the regions above shorter (top-aligned) panels with the background.
    fn fill_gaps(rgba: &mut [u8], total_width: u32, height: u32, panels: &[&Panel]) {
        let mut x_offset = 0u32;
        for panel in panels {
            for y in panel.height..height {
                for x in 0..panel.width {
                    let dest_x = x_offset + x;
                    let idx = ((y as usize) * (total_width as usize) + (dest_x as usize)) * 4;
                    rgba[idx..idx + 4].copy_from_slice(&BACKGROUND);
                }
            }
            x_offset += panel.width;
        }
    }

    /// The composited sheet width in pixels.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// The composited sheet height in pixels.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Encode the sheet as a deterministic RGBA8 PNG.
    ///
    /// Returns `None` only if the composited raster is degenerate (zero area),
    /// which a non-empty panel pair cannot produce.
    #[must_use]
    pub fn encode_png(&self) -> Option<Vec<u8>> {
        encode_rgba(self.width, self.height, &self.rgba)
    }
}

#[cfg(test)]
mod tests {
    use super::{BACKGROUND, ContactSheet, Panel};

    fn solid(width: u32, height: u32, color: [u8; 4]) -> Panel {
        let rgba = color
            .iter()
            .copied()
            .cycle()
            .take((width as usize) * (height as usize) * 4)
            .collect();
        Panel::new(width, height, rgba).expect("solid panel")
    }

    #[test]
    fn panel_rejects_inconsistent_buffer() {
        assert!(Panel::new(2, 2, vec![0; 15]).is_none());
        assert!(Panel::new(0, 2, vec![]).is_none());
    }

    #[test]
    fn sheet_lays_panels_side_by_side() {
        let before = solid(4, 4, [255, 0, 0, 255]);
        let after = solid(4, 4, [0, 255, 0, 255]);
        let sheet = ContactSheet::compose(&before, &after);
        // before + after + diff, each 4 wide, all 4 tall.
        assert_eq!(sheet.width(), 12);
        assert_eq!(sheet.height(), 4);
        let png = sheet.encode_png().expect("encode");
        assert!(png.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn diff_amplifies_aligned_change_and_is_opaque() {
        // before black, after has a single bright pixel; diff must show it.
        let before = solid(2, 1, [0, 0, 0, 255]);
        let after = Panel::new(2, 1, vec![0, 0, 0, 255, 8, 0, 0, 255]).expect("after");
        let sheet = ContactSheet::compose(&before, &after);
        // Diff panel is the rightmost 2 columns. Pixel (1,0) of the diff: red
        // delta 8 * gain 8 = 64.
        // Sheet is 6 wide; diff starts at x=4. Diff pixel (1,0) -> sheet (5,0).
        let idx = (5 * 4) as usize;
        assert_eq!(sheet.rgba[idx], 64, "amplified red delta");
        assert_eq!(sheet.rgba[idx + 3], 255, "diff alpha is opaque");
    }

    #[test]
    fn mismatched_panel_sizes_still_compose_rectangularly() {
        let before = solid(4, 4, [10, 10, 10, 255]);
        let after = solid(2, 2, [20, 20, 20, 255]);
        let sheet = ContactSheet::compose(&before, &after);
        // Width = 4 + 2 + diff(max width = 4) = 10; height = max(4,2,4) = 4.
        assert_eq!(sheet.width(), 10);
        assert_eq!(sheet.height(), 4);
        // The area above the shorter `after` panel is background. `after` lives
        // at x in [4,6); rows 2..4 above it are background.
        let idx = ((2 * 10 + 4) * 4) as usize;
        assert_eq!(&sheet.rgba[idx..idx + 4], &BACKGROUND);
    }

    #[test]
    fn composition_is_byte_deterministic() {
        let before = solid(3, 3, [1, 2, 3, 255]);
        let after = solid(3, 3, [9, 8, 7, 255]);
        let a = ContactSheet::compose(&before, &after)
            .encode_png()
            .expect("a");
        let b = ContactSheet::compose(&before, &after)
            .encode_png()
            .expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn composited_sheet_decodes_as_a_real_png() {
        let before = solid(2, 2, [200, 100, 50, 255]);
        let after = solid(2, 2, [50, 100, 200, 255]);
        let png = ContactSheet::compose(&before, &after)
            .encode_png()
            .expect("encode");
        let decoded = image::load_from_memory(&png).expect("decode");
        assert_eq!(decoded.width(), 6);
        assert_eq!(decoded.height(), 2);
    }
}
