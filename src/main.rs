use clap::Parser;
use fidget::{
    types::{Grad, Interval},
    vm::Choice,
};
use image::{ImageBuffer, Rgba};
use std::{collections::HashMap, path::PathBuf};

#[derive(Copy, Clone, Debug)]
enum Op {
    VarX,
    VarY,
    Copy(u32),
    Const(f32),
    Square(u32),
    Neg(u32),
    Sqrt(u32),
    Add(u32, u32),
    Sub(u32, u32),
    Div(u32, u32),
    Mul(u32, u32),
    Max(u32, u32),
    Min(u32, u32),
}

fn parse(s: &str) -> Vec<(u32, Op)> {
    let mut ops = vec![];
    let mut slots = HashMap::new();
    let mut get_slot = |arg: &str| {
        let i = u32::try_from(slots.len()).unwrap();
        *slots.entry(arg.to_owned()).or_insert(i)
    };
    for line in s.lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut iter = line.split_whitespace();
        let out = get_slot(iter.next().unwrap());
        let opcode = iter.next().unwrap();
        let op = match opcode {
            "const" => {
                let f = iter.next().unwrap().parse::<f32>().unwrap();
                Op::Const(f)
            }
            "var-x" => Op::VarX,
            "var-y" => Op::VarY,
            "add" | "sub" | "mul" | "max" | "min" | "div" => {
                let a = get_slot(iter.next().unwrap());
                let b = get_slot(iter.next().unwrap());
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
                let a = get_slot(iter.next().unwrap());
                match opcode {
                    "square" => Op::Square(a),
                    "neg" => Op::Neg(a),
                    "sqrt" => Op::Sqrt(a),
                    _ => unreachable!(),
                }
            }
            _ => panic!("unknown opcode {opcode}"),
        };
        ops.push((out, op));
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

fn pixel_f<const N: usize>(
    ops: &[(u32, Op)],
    scratch: &mut Vec<[f32; N]>,
    x: [f32; N],
    y: [f32; N],
) -> [f32; N] {
    scratch.resize(
        ops.last().map(|(i, _)| *i as usize).unwrap_or(0) + 1,
        [f32::NAN; N],
    );
    for (i, op) in ops {
        let i = *i as usize;
        match op {
            Op::VarX => scratch[i] = x,
            Op::VarY => scratch[i] = y,
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
                    assert!(!scratch[*a as usize][j].is_nan());
                    assert!(!scratch[*b as usize][j].is_nan());
                    scratch[i][j] =
                        scratch[*a as usize][j].max(scratch[*b as usize][j])
                }
            }
            Op::Min(a, b) => {
                for j in 0..N {
                    assert!(!scratch[*a as usize][j].is_nan());
                    assert!(!scratch[*b as usize][j].is_nan());
                    scratch[i][j] =
                        scratch[*a as usize][j].min(scratch[*b as usize][j])
                }
            }
            Op::Const(f) => {
                for j in 0..N {
                    scratch[i][j] = *f;
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
    scratch.last().cloned().unwrap_or([0.0; N])
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
    ops: &[(u32, Op)],
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
                    scratch[i][j] = Grad::from(*f)
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
    scratch.last().cloned().unwrap_or([Grad::from(0.0); N])
}

fn interval_i<const N: usize>(
    ops: &[(u32, Op)],
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
                    scratch[i][j] = Interval::from(*f)
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
    (
        scratch.last().cloned().unwrap_or([Interval::from(0.0); N]),
        choices,
    )
}

fn simplify<const N: usize>(
    ops: &[(u32, Op)],
    choices: &[[Choice; N]],
) -> [Vec<(u32, Op)>; N] {
    let mut out = [(); N].map(|()| vec![]);
    let mut active = vec![[0; N]; ops.len()];
    let mut choices = choices.iter().rev();
    if let Some(a) = active.last_mut() {
        *a = [1; N];
    }
    let push_copy = |v: &mut Vec<(u32, Op)>, allowed: bool, i: u32, a: u32| {
        if let Some((_i, Op::Copy(prev))) = v.last_mut() {
            if allowed && *prev == i {
                *prev = a;
                return;
            }
        }
        v.push((i, Op::Copy(a)));
    };
    for (i, o) in ops.iter().rev() {
        match o {
            Op::Const(..) | Op::VarX | Op::VarY => {
                for (j, v) in out.iter_mut().enumerate() {
                    if active[*i as usize][j] > 0 {
                        v.push((*i, *o))
                    }
                }
            }
            Op::Copy(a) => {
                for (j, v) in out.iter_mut().enumerate() {
                    if active[*i as usize][j] > 0 {
                        push_copy(v, active[*i as usize][j] == 1, *i, *a);
                    }
                }
            }
            Op::Sqrt(a) | Op::Square(a) | Op::Neg(a) => {
                for (j, v) in out.iter_mut().enumerate() {
                    if active[*i as usize][j] > 0 {
                        active[*a as usize][j] += 1;
                        v.push((*i, *o));
                    }
                }
            }
            Op::Add(a, b) | Op::Sub(a, b) | Op::Mul(a, b) | Op::Div(a, b) => {
                for (j, v) in out.iter_mut().enumerate() {
                    if active[*i as usize][j] > 0 {
                        active[*a as usize][j] += 1;
                        active[*b as usize][j] += 1;
                        v.push((*i, *o));
                    }
                }
            }
            Op::Min(a, b) | Op::Max(a, b) => {
                let cs = choices.next().unwrap();
                for (j, v) in out.iter_mut().enumerate() {
                    if active[*i as usize][j] > 0 {
                        match cs[j] {
                            Choice::Left => {
                                active[*a as usize][j] += 1;
                                push_copy(
                                    v,
                                    active[*i as usize][j] == 1,
                                    *i,
                                    *a,
                                );
                            }
                            Choice::Right => {
                                active[*b as usize][j] += 1;
                                push_copy(
                                    v,
                                    active[*i as usize][j] == 1,
                                    *i,
                                    *b,
                                );
                            }
                            Choice::Both => {
                                active[*a as usize][j] += 1;
                                active[*b as usize][j] += 1;
                                v.push((*i, *o))
                            }
                            Choice::Unknown => panic!(),
                        }
                    }
                }
            }
        }
    }
    assert!(choices.next().is_none());
    for a in &mut out {
        a.reverse()
    }
    out
}

fn shade(f: f32) -> [u8; 4] {
    use fidget::render::RenderMode;
    let [r, g, b] = fidget::render::SdfPixelRenderMode::pixel(f);
    [r, g, b, 255]
}

fn render_multistage(ops: &[(u32, Op)], size: u32) -> Vec<[u8; 4]> {
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

    let mut scratch_i = vec![];
    let mut scratch_f = vec![];
    let mut pixels = vec![[0u8; 4]; (size as usize).pow(2)];
    for ts in tiles.chunks(INTERVAL_SIMD_SIZE as usize) {
        let mut xs = [Interval::from(0.0); INTERVAL_SIMD_SIZE as usize];
        let mut ys = [Interval::from(0.0); INTERVAL_SIMD_SIZE as usize];
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
        let next = simplify(ops, &choices);
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
                    for ix in 0..TILE_SIZE {
                        pixels[(ix
                            + ts[j].x
                            + (size - (iy + ts[j].y) - 1) * size)
                            as usize] = fill;
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
                    let vs = pixel_f(&next[j], &mut scratch_f, xs, ys);
                    for k in 0..FLOAT_SIMD_SIZE as usize {
                        pixels[(ix * FLOAT_SIMD_SIZE
                            + ts[j].x
                            + k as u32
                            + (size - (iy + ts[j].y) - 1) * size)
                            as usize] = shade(vs[k]);
                    }
                }
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
    let mut scratch_f = vec![];
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
            pixels = render_multistage(&ops, args.size);
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
