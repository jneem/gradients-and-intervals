use clap::Parser;
use fidget::{
    types::{Grad, Interval},
    vm::Choice,
};
use image::{ImageBuffer, Rgba};
use rayon::prelude::*;
use std::{collections::HashMap, path::PathBuf};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
enum Op {
    VarX,
    VarY,
    Copy(u16),
    Const(ordered_float::OrderedFloat<f32>),
    Square(u16),
    Neg(u16),
    Sqrt(u16),
    Add(u16, u16),
    Sub(u16, u16),
    Div(u16, u16),
    Mul(u16, u16),
    Max(u16, u16),
    Min(u16, u16),
}

fn parse(s: &str) -> Vec<(u16, Op)> {
    let mut ops = vec![];

    #[derive(Default)]
    struct Slots<'a>(HashMap<&'a str, u16>);
    impl<'a> Slots<'a> {
        fn get(&mut self, arg: &'a str) -> u16 {
            let i = u16::try_from(self.0.len()).unwrap();
            *self.0.entry(arg).or_insert(i)
        }
        fn put(&mut self, s: &'a str, v: u16) {
            let prev = self.0.insert(s, v);
            assert!(prev.is_none());
        }
    }

    let mut slots = Slots::default();
    let mut seen = HashMap::new();
    for line in s.lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut iter = line.split_whitespace();
        let out = iter.next().unwrap();
        let opcode = iter.next().unwrap();
        let op = match opcode {
            "const" => {
                let f = iter.next().unwrap().parse::<f32>().unwrap();
                Op::Const(f.into())
            }
            "var-x" => Op::VarX,
            "var-y" => Op::VarY,
            "add" | "sub" | "mul" | "max" | "min" | "div" => {
                let a = slots.get(iter.next().unwrap());
                let b = slots.get(iter.next().unwrap());
                match opcode {
                    "add" => Op::Add(a, b),
                    "sub" => Op::Sub(a, b),
                    "mul" => Op::Mul(a, b),
                    "min" => Op::Min(a, b),
                    "max" => Op::Max(a, b),
                    "div" => Op::Div(a, b),
                    _ => unreachable!(),
                }
            }
            "square" | "neg" | "sqrt" => {
                let a = slots.get(iter.next().unwrap());
                match opcode {
                    "square" => Op::Square(a),
                    "neg" => Op::Neg(a),
                    "sqrt" => Op::Sqrt(a),
                    _ => unreachable!(),
                }
            }
            _ => panic!("unknown opcode {opcode}"),
        };
        let prev = match seen.entry(op) {
            std::collections::hash_map::Entry::Occupied(e) => {
                slots.put(out, *e.get());
                *e.get()
            }
            std::collections::hash_map::Entry::Vacant(e) => {
                let i = slots.get(out);
                e.insert(i);
                i
            }
        };
        ops.push((prev, op));
    }
    ops
}

#[derive(clap::Parser)]
struct Args {
    /// Name of a VM file to read
    #[clap(short, long)]
    input: Option<PathBuf>,

    /// Name of a `.png` file to write
    #[clap(short, long)]
    out: Option<PathBuf>,

    #[clap(short, long, default_value_t = 512)]
    size: u32,

    #[clap(long)]
    normalize: bool,

    #[clap(long)]
    fancy: bool,

    #[clap(short = 'N', default_value_t = 1)]
    count: u32,
}

struct Slots<T, const N: usize> {
    slots: Vec<[T; N]>,
}

impl<T: Copy + Clone + From<f32>, const N: usize> Slots<T, N> {
    fn new() -> Self {
        Self { slots: vec![] }
    }
    fn resize(&mut self, capacity: usize) {
        self.slots.resize(capacity, [T::from(f32::NAN); N])
    }
}

impl<T, const N: usize> std::ops::Index<u16> for Slots<T, N> {
    type Output = [T; N];
    fn index(&self, index: u16) -> &Self::Output {
        &self.slots[index as usize]
    }
}

impl<T, const N: usize> std::ops::IndexMut<u16> for Slots<T, N> {
    fn index_mut(&mut self, index: u16) -> &mut Self::Output {
        &mut self.slots[index as usize]
    }
}

fn pixel_f<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Slots<f32, N>,
    x: [f32; N],
    y: [f32; N],
) -> [f32; N] {
    scratch.resize(ops.len());
    for (i, op) in ops {
        let i = *i;
        match op {
            Op::VarX => scratch[i] = x,
            Op::VarY => scratch[i] = y,
            Op::Copy(a) => scratch[i] = scratch[*a],
            Op::Add(a, b) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a][j] + scratch[*b][j]
                }
            }
            Op::Sub(a, b) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a][j] - scratch[*b][j]
                }
            }
            Op::Mul(a, b) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a][j] * scratch[*b][j]
                }
            }
            Op::Div(a, b) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a][j] / scratch[*b][j]
                }
            }
            Op::Max(a, b) => {
                for j in 0..N {
                    assert!(!scratch[*a][j].is_nan());
                    assert!(!scratch[*b][j].is_nan());
                    scratch[i][j] = scratch[*a][j].max(scratch[*b][j])
                }
            }
            Op::Min(a, b) => {
                for j in 0..N {
                    assert!(!scratch[*a][j].is_nan());
                    assert!(!scratch[*b][j].is_nan());
                    scratch[i][j] = scratch[*a][j].min(scratch[*b][j])
                }
            }
            Op::Const(f) => {
                for j in 0..N {
                    scratch[i][j] = **f;
                }
            }
            Op::Square(a) => {
                for j in 0..N {
                    let v = scratch[*a][j];
                    scratch[i][j] = v * v;
                }
            }
            Op::Neg(a) => {
                for j in 0..N {
                    scratch[i][j] = -scratch[*a][j];
                }
            }
            Op::Sqrt(a) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a][j].sqrt();
                }
            }
        };
    }
    scratch[ops.last().unwrap().0]
}

/// Normalizes to a gradient of 1
///
/// If the gradient is very small, then the value is scaled up based on the
/// internal epsilon parameter
pub fn normalize(g: Grad) -> Grad {
    let norm = (g.dx.powi(2) + g.dy.powi(2) + g.dz.powi(2))
        .sqrt()
        .max(0.01);
    Grad::new(g.v / norm, g.dx / norm, g.dy / norm, g.dz / norm)
}

fn pixel_g<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Vec<[Grad; N]>,
    x: [f32; N],
    y: [f32; N],
) -> [Grad; N] {
    scratch.resize(ops.len(), [Grad::from(0.0); N]);
    for (i, op) in ops {
        let i = *i as usize;
        match op {
            Op::VarX => {
                for (j, x) in x.iter().enumerate() {
                    scratch[i][j] = Grad::new(*x, 1.0, 0.0, 0.0)
                }
            }
            Op::VarY => {
                for (j, y) in y.iter().enumerate() {
                    scratch[i][j] = Grad::new(*y, 0.0, 1.0, 0.0)
                }
            }
            Op::Copy(a) => scratch[i] = scratch[*a as usize],
            Op::Add(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] + scratch[*b as usize][j]
                }
            }
            Op::Sub(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] - scratch[*b as usize][j]
                }
            }
            Op::Mul(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] * scratch[*b as usize][j]
                }
            }
            Op::Div(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] / scratch[*b as usize][j]
                }
            }
            Op::Max(a, b) => {
                for j in 0..N {
                    scratch[i][j] = normalize(scratch[*a as usize][j])
                        .max(normalize(scratch[*b as usize][j]))
                }
            }
            Op::Min(a, b) => {
                for j in 0..N {
                    scratch[i][j] = normalize(scratch[*a as usize][j])
                        .min(normalize(scratch[*b as usize][j]))
                }
            }
            Op::Const(f) => {
                for j in 0..N {
                    scratch[i][j] = Grad::from(**f)
                }
            }
            Op::Square(a) => {
                for j in 0..N {
                    let v = scratch[*a as usize][j];
                    scratch[i][j] = v * v;
                }
            }
            Op::Neg(a) => {
                for j in 0..N {
                    scratch[i][j] = -scratch[*a as usize][j];
                }
            }
            Op::Sqrt(a) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a as usize][j].sqrt();
                }
            }
        };
    }
    scratch[ops.last().unwrap().0 as usize]
}

fn float_tile<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Slots<f32, N>,
    x: Interval,
    y: Interval,
) -> [f32; N] {
    let side = (N as f64).sqrt() as usize;
    assert_eq!(side * side, N);
    let mut xs = [0.0; N];
    let mut ys = [0.0; N];
    let mut k = 0;
    for i in 0..side {
        let y = y.lerp(i as f32 / side as f32);
        for j in 0..side {
            let x = x.lerp(j as f32 / side as f32);
            xs[k] = x;
            ys[k] = y;
            k += 1;
        }
    }
    pixel_f(ops, scratch, xs, ys)
}

fn interval_split<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Vec<[Interval; N]>,
    x: Interval,
    y: Interval,
) -> ([Interval; N], Vec<[Choice; N]>) {
    let side = (N as f64).sqrt() as usize;
    assert_eq!(side * side, N);
    let mut xs = [Interval::from(0.0); N];
    let mut ys = [Interval::from(0.0); N];
    let mut k = 0;
    for i in 0..side {
        let y = Interval::new(
            y.lerp(i as f32 / side as f32),
            y.lerp((i + 1) as f32 / side as f32),
        );
        for j in 0..side {
            let x = Interval::new(
                x.lerp(j as f32 / side as f32),
                x.lerp((j + 1) as f32 / side as f32),
            );
            xs[k] = x;
            ys[k] = y;
            k += 1;
        }
    }
    interval_i(ops, scratch, xs, ys)
}

fn interval_i<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Vec<[Interval; N]>,
    x: [Interval; N],
    y: [Interval; N],
) -> ([Interval; N], Vec<[Choice; N]>) {
    scratch.resize(ops.len(), [Interval::from(0.0); N]);
    let mut choices = vec![];
    for (i, op) in ops {
        let i = *i as usize;
        match op {
            Op::VarX => {
                scratch[i] = x;
            }
            Op::VarY => {
                scratch[i] = y;
            }
            Op::Copy(a) => scratch[i] = scratch[*a as usize],
            Op::Add(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] + scratch[*b as usize][j]
                }
            }
            Op::Sub(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] - scratch[*b as usize][j]
                }
            }
            Op::Mul(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] * scratch[*b as usize][j]
                }
            }
            Op::Div(a, b) => {
                for j in 0..N {
                    scratch[i][j] =
                        scratch[*a as usize][j] / scratch[*b as usize][j]
                }
            }
            Op::Max(a, b) => {
                let mut cs = [Choice::Both; N];
                for (j, cs) in cs.iter_mut().enumerate() {
                    let (v, c) = scratch[*a as usize][j]
                        .max_choice(scratch[*b as usize][j]);
                    scratch[i][j] = v;
                    *cs = c;
                }
                choices.push(cs);
            }
            Op::Min(a, b) => {
                let mut cs = [Choice::Both; N];
                for (j, cs) in cs.iter_mut().enumerate() {
                    let (v, c) = scratch[*a as usize][j]
                        .min_choice(scratch[*b as usize][j]);
                    scratch[i][j] = v;
                    *cs = c;
                }
                choices.push(cs);
            }
            Op::Const(f) => {
                for j in 0..N {
                    scratch[i][j] = Interval::from(**f)
                }
            }
            Op::Square(a) => {
                for j in 0..N {
                    let v = scratch[*a as usize][j];
                    scratch[i][j] = v.square();
                }
            }
            Op::Neg(a) => {
                for j in 0..N {
                    scratch[i][j] = -scratch[*a as usize][j];
                }
            }
            Op::Sqrt(a) => {
                for j in 0..N {
                    scratch[i][j] = scratch[*a as usize][j].sqrt();
                }
            }
        };
    }
    (scratch[ops.last().unwrap().0 as usize], choices)
}

struct Simplify<const N: usize> {
    active: Vec<[Option<u16>; N]>,
    out: [Vec<(u16, Op)>; N],
    next: [u16; N],
}

impl<const N: usize> Simplify<N> {
    fn new() -> Self {
        Self {
            active: vec![],
            out: [(); N].map(|()| vec![]),
            next: [0; N],
        }
    }

    /// Bind a register to the next available slot
    fn bind(&mut self, index: usize, a: u16) -> u16 {
        *self.active[a as usize][index].get_or_insert_with(|| {
            let v = self.next[index];
            self.next[index] += 1;
            v
        })
    }

    fn simplify_unary(&mut self, i: u16, a: u16, f: fn(u16) -> Op) {
        for j in 0..N {
            if let Some(i) = self.active[i as usize][j] {
                let a = self.bind(j, a);
                self.out[j].push((i, f(a)));
            }
        }
    }

    fn simplify_binary(
        &mut self,
        i: u16,
        a: u16,
        b: u16,
        f: fn(u16, u16) -> Op,
    ) {
        for j in 0..N {
            if let Some(i) = self.active[i as usize][j] {
                let a = self.bind(j, a);
                let b = self.bind(j, b);
                self.out[j].push((i, f(a, b)));
            }
        }
    }

    fn simplify_binary_choice(
        &mut self,
        i: u16,
        a: u16,
        b: u16,
        cs: [Choice; N],
        f: fn(u16, u16) -> Op,
    ) {
        for (j, c) in cs.iter().enumerate() {
            if let Some(i) = self.active[i as usize][j] {
                match *c {
                    Choice::Left => {
                        self.push_copy(j, i, a);
                    }
                    Choice::Right => {
                        self.push_copy(j, i, b);
                    }
                    Choice::Both => {
                        let a = self.bind(j, a);
                        let b = self.bind(j, b);
                        self.out[j].push((i, f(a, b)));
                    }
                    Choice::Unknown => panic!(),
                }
            }
        }
    }

    fn push_copy(&mut self, index: usize, i: u16, a: u16) {
        if let Some(a) = self.active[a as usize][index] {
            self.out[index].push((i, Op::Copy(a)));
        } else {
            self.active[a as usize][index] = Some(i);
        }
    }

    fn run(&mut self, ops: &[(u16, Op)], choices: &[[Choice; N]]) {
        self.active.fill([None; N]);
        self.active.resize(ops.len(), [None; N]);
        self.next = [0; N];
        self.out = [(); N].map(|()| Vec::with_capacity(ops.len() / 2));

        let mut choices = choices.iter().rev();
        if let Some((i, _op)) = ops.last() {
            self.active[*i as usize] = [Some(0); N];
            self.next = [1; N];
        }

        for (i, o) in ops.iter().rev() {
            match o {
                Op::Const(..) | Op::VarX | Op::VarY => {
                    for (j, v) in self.out.iter_mut().enumerate() {
                        if let Some(i) = self.active[*i as usize][j] {
                            v.push((i, *o))
                        }
                    }
                }
                Op::Copy(a) => self.simplify_unary(*i, *a, Op::Copy),
                Op::Sqrt(a) => self.simplify_unary(*i, *a, Op::Sqrt),
                Op::Square(a) => self.simplify_unary(*i, *a, Op::Square),
                Op::Neg(a) => self.simplify_unary(*i, *a, Op::Neg),
                Op::Add(a, b) => self.simplify_binary(*i, *a, *b, Op::Add),
                Op::Sub(a, b) => self.simplify_binary(*i, *a, *b, Op::Sub),
                Op::Mul(a, b) => self.simplify_binary(*i, *a, *b, Op::Mul),
                Op::Div(a, b) => self.simplify_binary(*i, *a, *b, Op::Div),
                Op::Min(a, b) => self.simplify_binary_choice(
                    *i,
                    *a,
                    *b,
                    *choices.next().unwrap(),
                    Op::Min,
                ),
                Op::Max(a, b) => self.simplify_binary_choice(
                    *i,
                    *a,
                    *b,
                    *choices.next().unwrap(),
                    Op::Max,
                ),
            }
        }
        assert!(choices.next().is_none());

        for a in self.out.iter_mut() {
            a.reverse();
        }
    }
}

fn simplify<const N: usize>(
    ops: &[(u16, Op)],
    choices: &[[Choice; N]],
) -> [Vec<(u16, Op)>; N] {
    let mut worker = Simplify::new();
    worker.run(ops, choices);
    worker.out
}

fn shade(f: f32) -> [u8; 4] {
    use fidget::render::RenderMode;
    let [r, g, b] = fidget::render::SdfPixelRenderMode::pixel(f);
    [r, g, b, 255]
}

fn render_recursive(ops: &[(u16, Op)], size: u32) -> Vec<[u8; 4]> {
    assert_eq!(size % 512, 0, "size {size} must be divisible by 512");
    let mut scratch_i = vec![];
    let mut scratch_f = Slots::new();
    let bounds = Interval::new(-1.0, 1.0);
    let mut pixels = vec![[0u8; 4]; (size as usize).pow(2)];
    let mut fill = |x: u32, y: u32, tile_size: u32, color| {
        for j in 0..tile_size {
            for i in 0..tile_size {
                let k = (size - (y + j) - 1) * size + x + i;
                pixels[k as usize] = color;
            }
        }
    };

    // Render a set of 128x128 interval regions
    let mut next = vec![];
    for x in 0..size / 512 {
        let x = x * 512;
        for y in 0..size / 512 {
            let y = y * 512;
            let ix = Interval::new(
                bounds.lerp(x as f32 / size as f32),
                bounds.lerp((x + 512) as f32 / size as f32),
            );
            let iy = Interval::new(
                bounds.lerp(y as f32 / size as f32),
                bounds.lerp((y + 512) as f32 / size as f32),
            );
            let (values, choices) =
                interval_split::<16>(ops, &mut scratch_i, ix, iy);
            let mut simplified = simplify(ops, &choices);
            for j in 0..4 {
                for i in 0..4 {
                    let k = j * 4 + i;
                    if values[k].contains(0.0) {
                        next.push((
                            x + i as u32 * 128,
                            y + j as u32 * 128,
                            std::mem::take(&mut simplified[k]),
                        ));
                    } else {
                        fill(
                            x + i as u32 * 128,
                            y + j as u32 * 128,
                            128,
                            [128, 128, 128, 255],
                        );
                    }
                }
            }
        }
    }

    // Render 32x32 interval regions
    let tiles = next;
    let mut next = vec![];
    for (x, y, tape) in tiles {
        let ix = Interval::new(
            bounds.lerp(x as f32 / size as f32),
            bounds.lerp((x + 128) as f32 / size as f32),
        );
        let iy = Interval::new(
            bounds.lerp(y as f32 / size as f32),
            bounds.lerp((y + 128) as f32 / size as f32),
        );
        let (values, choices) =
            interval_split::<16>(&tape, &mut scratch_i, ix, iy);
        let mut simplified = simplify(&tape, &choices);
        for j in 0..4 {
            for i in 0..4 {
                let k = j * 4 + i;
                if values[k].contains(0.0) {
                    next.push((
                        x + i as u32 * 32,
                        y + j as u32 * 32,
                        std::mem::take(&mut simplified[k]),
                    ));
                } else {
                    fill(
                        x + i as u32 * 32,
                        y + j as u32 * 32,
                        32,
                        [255, 128, 128, 255],
                    );
                }
            }
        }
    }

    // Render 8x8 interval regions
    let tiles = next;
    let mut next = vec![];
    for (x, y, tape) in tiles {
        let ix = Interval::new(
            bounds.lerp(x as f32 / size as f32),
            bounds.lerp((x + 32) as f32 / size as f32),
        );
        let iy = Interval::new(
            bounds.lerp(y as f32 / size as f32),
            bounds.lerp((y + 32) as f32 / size as f32),
        );
        let (values, choices) =
            interval_split::<16>(&tape, &mut scratch_i, ix, iy);
        let mut simplified = simplify(&tape, &choices);
        for j in 0..4 {
            for i in 0..4 {
                let k = j * 4 + i;
                if values[k].contains(0.0) {
                    next.push((
                        x + i as u32 * 8,
                        y + j as u32 * 8,
                        std::mem::take(&mut simplified[k]),
                    ));
                } else {
                    fill(
                        x + i as u32 * 8,
                        y + j as u32 * 8,
                        8,
                        [255, 255, 128, 255],
                    );
                }
            }
        }
    }

    // Render 8x8 tiles
    let tiles = next;
    let mut next = vec![];
    for (x, y, tape) in tiles {
        let ix = Interval::new(
            bounds.lerp(x as f32 / size as f32),
            bounds.lerp((x + 8) as f32 / size as f32),
        );
        let iy = Interval::new(
            bounds.lerp(y as f32 / size as f32),
            bounds.lerp((y + 8) as f32 / size as f32),
        );
        let vs = float_tile::<64>(&tape, &mut scratch_f, ix, iy);
        next.push((x, y, vs));
    }

    for (x, y, vs) in next {
        for j in 0..8 {
            for i in 0..8 {
                let k = j * 8 + i;
                let p = (size - (y + j) - 1) * size + x + i;
                pixels[p as usize] = shade(vs[k as usize]);
            }
        }
    }
    pixels
}

fn render_multistage(ops: &[(u16, Op)], size: u32) -> Vec<[u8; 4]> {
    const TILE_SIZE: u32 = 32;
    const INTERVAL_SIMD_SIZE: u32 = 16;
    const FLOAT_SIMD_SIZE: u32 = 32;
    assert_eq!(
        TILE_SIZE % FLOAT_SIMD_SIZE,
        0,
        "chunk size must be divisible by SIMD size"
    );

    #[derive(Debug)]
    struct Tile {
        x: u32,
        y: u32,
    }
    assert_eq!(
        size % INTERVAL_SIMD_SIZE,
        0,
        "image size must be divisible by {INTERVAL_SIMD_SIZE}"
    );
    let mut tiles = vec![];
    for x in 0..size / TILE_SIZE {
        for y in 0..size / TILE_SIZE {
            tiles.push(Tile {
                x: x * TILE_SIZE,
                y: y * TILE_SIZE,
            });
        }
    }
    assert_eq!(
        tiles.len() % INTERVAL_SIMD_SIZE as usize,
        0,
        "tile count must be divisible by {INTERVAL_SIMD_SIZE}"
    );

    let to_image_pos = |i: u32| -> f32 {
        ((i as f32 - size as f32 / 2.0) / size as f32) * 2.0
    };

    let out = tiles
        .chunks(INTERVAL_SIMD_SIZE as usize)
        .map(|ts| {
            let mut out = vec![];
            let mut xs = [Interval::from(0.0); INTERVAL_SIMD_SIZE as usize];
            let mut ys = [Interval::from(0.0); INTERVAL_SIMD_SIZE as usize];
            let mut scratch_i = vec![];
            let mut scratch_f = Slots::new();
            for j in 0..INTERVAL_SIMD_SIZE as usize {
                xs[j] = Interval::new(
                    to_image_pos(ts[j].x),
                    to_image_pos(ts[j].x + TILE_SIZE),
                );
                ys[j] = Interval::new(
                    to_image_pos(ts[j].y),
                    to_image_pos(ts[j].y + TILE_SIZE),
                );
            }
            let (values, choices) = interval_i(ops, &mut scratch_i, xs, ys);
            let next = if values.iter().any(|v| v.contains(0.0)) {
                Some(simplify(ops, &choices))
            } else {
                None
            };
            for j in 0..INTERVAL_SIMD_SIZE as usize {
                let fill = if values[j].upper() < 0.0 {
                    Some([255, 0, 0, 255])
                } else if values[j].lower() > 0.0 {
                    Some([0, 255, 0, 255])
                } else {
                    None
                };
                if let Some(fill) = fill {
                    for iy in 0..TILE_SIZE {
                        for ix in 0..TILE_SIZE / FLOAT_SIMD_SIZE {
                            out.push((
                                ix * FLOAT_SIMD_SIZE
                                    + ts[j].x
                                    + (size - (iy + ts[j].y) - 1) * size,
                                [fill; FLOAT_SIMD_SIZE as usize],
                            ))
                        }
                    }
                    continue;
                }
                for iy in 0..TILE_SIZE {
                    let y = to_image_pos(iy + ts[j].y);
                    let ys = [y; FLOAT_SIMD_SIZE as usize];
                    for ix in 0..TILE_SIZE / FLOAT_SIMD_SIZE {
                        let mut xs = [0.0; FLOAT_SIMD_SIZE as usize];
                        for (k, x) in xs.iter_mut().enumerate() {
                            *x = to_image_pos(
                                ix * FLOAT_SIMD_SIZE + ts[j].x + k as u32,
                            );
                        }
                        let vs = pixel_f(
                            &next.as_ref().unwrap()[j],
                            &mut scratch_f,
                            xs,
                            ys,
                        );
                        out.push((
                            ix * FLOAT_SIMD_SIZE
                                + ts[j].x
                                + (size - (iy + ts[j].y) - 1) * size,
                            vs.map(shade),
                        ));
                    }
                }
            }
            out.into_iter()
        })
        .collect::<Vec<_>>();

    let mut pixels = vec![[0u8; 4]; (size as usize).pow(2)];
    for (i, vs) in out.into_iter().flatten() {
        pixels[i as usize..][..vs.len()].copy_from_slice(&vs);
    }
    pixels
}

fn main() {
    let args = Args::parse();
    let ops = match args.input {
        Some(p) => {
            let s = std::fs::read_to_string(p).expect("failed to read file");
            parse(&s)
        }
        None => parse(PROSPERO),
    };

    let mut pixels = Vec::with_capacity((args.size as usize).pow(2));
    let mut scratch_f = Slots::new();
    let mut scratch_g = vec![];

    const WIDTH: usize = 16;
    assert_eq!(
        args.size % (WIDTH as u32),
        0,
        "image size must be a multiple of {WIDTH}"
    );

    let start_time = std::time::Instant::now();
    for _ in 0..args.count {
        if args.fancy {
            pixels = render_recursive(&ops, args.size);
        } else {
            pixels.clear();
            for iy in (0..args.size).rev() {
                let y = ((iy as f32) / (args.size as f32) - 0.5) * 2.0;
                let ys = [y; WIDTH];
                for ix in 0..args.size / WIDTH as u32 {
                    let mut xs = [0.0; WIDTH];
                    for (j, x) in xs.iter_mut().enumerate() {
                        *x = ((ix as f32 * WIDTH as f32 + j as f32)
                            / (args.size as f32)
                            - 0.5)
                            * 2.0;
                    }
                    let vs = if args.normalize {
                        let gs = pixel_g(&ops, &mut scratch_g, xs, ys);
                        gs.map(|g| g.v * 2.0) // condensed field lines
                    } else {
                        pixel_f(&ops, &mut scratch_f, xs, ys)
                    };
                    for v in vs {
                        pixels.push(shade(v));
                    }
                }
            }
        }
    }
    let elapsed = start_time.elapsed();
    println!(
        "Rendered {} in {:.2?} ({:.2?} / frame)",
        args.count,
        elapsed,
        elapsed / args.count
    );

    if let Some(out) = &args.out {
        let raw_pixels: Vec<u8> = pixels.into_iter().flatten().collect();
        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(args.size, args.size, raw_pixels)
                .expect("failed to create image buffer");
        img.save(out).expect("failed to save image");
    }
    println!("Hello, world!");
}

const PROSPERO: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/prospero.txt"));
