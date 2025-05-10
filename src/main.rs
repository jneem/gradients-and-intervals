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
    Const(ordered_float::OrderedFloat<f32>),
    Copy(u16),
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
    scale: f32,
) -> [f32; N] {
    scratch.resize(ops.len());
    for (i, op) in ops {
        let i = *i;
        match op {
            Op::VarX => scratch[i] = x.map(|x| x * scale),
            Op::VarY => scratch[i] = y.map(|y| y * scale),
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
    scale: f32,
) -> [Grad; N] {
    scratch.resize(ops.len(), [Grad::from(0.0); N]);
    for (i, op) in ops {
        let i = *i as usize;
        match op {
            Op::VarX => {
                for (j, x) in x.iter().enumerate() {
                    scratch[i][j] = Grad::new(*x * scale, 1.0, 0.0, 0.0)
                }
            }
            Op::VarY => {
                for (j, y) in y.iter().enumerate() {
                    scratch[i][j] = Grad::new(*y * scale, 0.0, 1.0, 0.0)
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

#[inline(never)]
fn float_tile<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Slots<f32, N>,
    x: Interval,
    y: Interval,
    scale: f32,
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
    pixel_f(ops, scratch, xs, ys, scale)
}

fn interval_split<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Vec<[Interval; N]>,
    x: Interval,
    y: Interval,
    scale: f32,
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
    interval_i(ops, scratch, xs, ys, scale)
}

fn interval_i<const N: usize>(
    ops: &[(u16, Op)],
    scratch: &mut Vec<[Interval; N]>,
    x: [Interval; N],
    y: [Interval; N],
    scale: f32,
) -> ([Interval; N], Vec<[Choice; N]>) {
    scratch.resize(ops.len(), [Interval::from(0.0); N]);
    let mut choices = vec![];
    for (i, op) in ops {
        let i = *i as usize;
        match op {
            Op::VarX => {
                scratch[i] = x.map(|x| x * scale);
            }
            Op::VarY => {
                scratch[i] = y.map(|y| y * scale);
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
                    println!("{} {}", v.square(), v * v);
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

struct Tile<'a> {
    x: u32,
    y: u32,
    tile_size: u32,
    image_size: u32,
    scale: f32,
    tape: std::borrow::Cow<'a, [(u16, Op)]>,
}

impl Tile<'_> {
    fn bounds(&self) -> (Interval, Interval) {
        let b = Interval::new(-1.0, 1.0);
        let ix = Interval::new(
            b.lerp(self.x as f32 / self.image_size as f32),
            b.lerp((self.x + self.tile_size) as f32 / self.image_size as f32),
        );
        let iy = Interval::new(
            b.lerp(self.y as f32 / self.image_size as f32),
            b.lerp((self.y + self.tile_size) as f32 / self.image_size as f32),
        );
        (ix, iy)
    }
}

struct Fill {
    x: u32,
    y: u32,
    tile_size: u32,
    image_size: u32,
    fill_color: [u8; 4],
    edge_color: [u8; 4],
}

impl Fill {
    fn run(&self, pixels: &mut [[u8; 4]]) {
        for j in 0..self.tile_size {
            for i in 0..self.tile_size {
                let k = (self.image_size - (self.y + j) - 1) * self.image_size
                    + self.x
                    + i;
                if i == 0
                    || i == self.tile_size - 1
                    || j == 0
                    || j == self.tile_size - 1
                {
                    pixels[k as usize] = self.edge_color;
                } else {
                    pixels[k as usize] = self.fill_color;
                }
            }
        }
    }
}

enum Next<'a> {
    Tile(Tile<'a>),
    Fill(Fill),
}

impl Tile<'_> {
    fn run(self, scratch: &mut Vec<[Interval; 16]>, out: &mut Vec<Next>) {
        let (ix, iy) = self.bounds();
        let (values, choices) =
            interval_split::<16>(&self.tape, scratch, ix, iy, self.scale);
        let mut simplified = simplify(&self.tape, &choices);
        for j in 0..4 {
            for i in 0..4 {
                let k = j * 4 + i;
                let x = self.x + i as u32 * self.tile_size / 4;
                let y = self.y + j as u32 * self.tile_size / 4;
                let tile_size = self.tile_size / 4;
                if values[k].contains(0.0) || values[k].has_nan() {
                    out.push(Next::Tile(Tile {
                        x,
                        y,
                        tile_size,
                        tape: std::borrow::Cow::Owned(std::mem::take(
                            &mut simplified[k],
                        )),
                        image_size: self.image_size,
                        scale: self.scale,
                    }));
                } else {
                    if tile_size == 128 {
                        println!("{ix} {iy} => {}", values[k]);
                    }
                    out.push(Next::Fill(Fill {
                        x,
                        y,
                        tile_size,
                        image_size: self.image_size,
                        fill_color: if values[k].upper() < 0.0 {
                            [200, 200, 200, 255]
                        } else {
                            [50, 50, 50, 255]
                        },
                        edge_color: if values[k].upper() < 0.0 {
                            [180, 180, 180, 255]
                        } else {
                            [20, 20, 20, 255]
                        },
                    }))
                }
            }
        }
    }
}

fn render_recursive(ops: &[(u16, Op)], size: u32, scale: f32) -> Vec<[u8; 4]> {
    assert_eq!(size % 512, 0, "size {size} must be divisible by 512");
    let mut scratch_i = vec![];
    let mut scratch_f = Slots::new();
    let mut pixels = vec![[0u8; 4]; (size as usize).pow(2)];

    // Render a set of 128x128 interval regions
    let mut tiles = vec![];
    for x in 0..size / 512 {
        let x = x * 512;
        for y in 0..size / 512 {
            let y = y * 512;
            tiles.push(Tile {
                x,
                y,
                tile_size: 512,
                image_size: size,
                tape: std::borrow::Cow::Borrowed(ops),
                scale,
            });
        }
    }

    // 512, 128, 32
    for _ in 0..3 {
        let mut next = vec![];
        for t in tiles {
            t.run(&mut scratch_i, &mut next);
        }
        tiles = vec![];
        for n in next {
            match n {
                Next::Tile(t) => tiles.push(t),
                Next::Fill(f) => f.run(&mut pixels),
            }
        }
    }

    let mut next = vec![];
    for t in tiles {
        let (ix, iy) = t.bounds();
        let vs = float_tile::<64>(&t.tape, &mut scratch_f, ix, iy, scale);
        next.push((t.x, t.y, t.tile_size, vs));
    }

    for (x, y, tile_size, vs) in next {
        for j in 0..tile_size {
            for i in 0..tile_size {
                let k = j * tile_size + i;
                let p = (size - (y + j) - 1) * size + x + i;
                pixels[p as usize] = if vs[k as usize] < 0.0 {
                    [255; 4]
                } else {
                    [0, 0, 0, 255]
                };
            }
        }
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

    let image_size = args.size.next_multiple_of(512);
    let scale = image_size as f32 / args.size as f32;

    let start_time = std::time::Instant::now();
    for _ in 0..args.count {
        if args.fancy {
            pixels = render_recursive(&ops, image_size, scale);
        } else {
            pixels.clear();
            for iy in (0..image_size).rev() {
                let y = ((iy as f32) / (image_size as f32) - 0.5) * 2.0;
                let ys = [y; WIDTH];
                for ix in 0..image_size / WIDTH as u32 {
                    let mut xs = [0.0; WIDTH];
                    for (j, x) in xs.iter_mut().enumerate() {
                        *x = ((ix as f32 * WIDTH as f32 + j as f32)
                            / (image_size as f32)
                            - 0.5)
                            * 2.0;
                    }
                    let vs = if args.normalize {
                        let gs = pixel_g(&ops, &mut scratch_g, xs, ys, scale);
                        gs.map(|g| g.v * 2.0) // condensed field lines
                    } else {
                        pixel_f(&ops, &mut scratch_f, xs, ys, scale)
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
        let mut raw_pixels = vec![[0u8; 4]; (args.size as usize).pow(2)];
        let offset = (image_size - args.size) / 2;
        for j in 0..args.size {
            for i in 0..args.size {
                raw_pixels[(j * args.size + i) as usize] =
                    pixels[((j + offset) * image_size + i + offset) as usize];
            }
        }
        let raw_pixels: Vec<u8> = raw_pixels.into_iter().flatten().collect();
        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(args.size, args.size, raw_pixels)
                .expect("failed to create image buffer");
        img.save(out).expect("failed to save image");
    }
    println!("Hello, world!");
}

const PROSPERO: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/prospero.txt"));
