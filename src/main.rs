use clap::Parser;
use fidget::eval::MathFunction;
use fidget::types::{Grad, Interval};
use image::{ImageBuffer, Rgba};
use std::{collections::HashMap, path::PathBuf};

enum Op {
    VarX,
    VarY,
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
}

fn pixel_f(ops: &[(u32, Op)], scratch: &mut Vec<f32>, x: f32, y: f32) -> f32 {
    scratch.resize(ops.len(), 0.0);
    for (i, op) in ops {
        let v = match op {
            Op::VarX => x,
            Op::VarY => y,
            Op::Add(a, b) => scratch[*a as usize] + scratch[*b as usize],
            Op::Sub(a, b) => scratch[*a as usize] - scratch[*b as usize],
            Op::Mul(a, b) => scratch[*a as usize] * scratch[*b as usize],
            Op::Div(a, b) => scratch[*a as usize] / scratch[*b as usize],
            Op::Max(a, b) => scratch[*a as usize].max(scratch[*b as usize]),
            Op::Min(a, b) => scratch[*a as usize].min(scratch[*b as usize]),
            Op::Const(f) => *f,
            Op::Square(a) => scratch[*a as usize].powi(2),
            Op::Neg(a) => -scratch[*a as usize],
            Op::Sqrt(a) => scratch[*a as usize].sqrt(),
        };
        scratch[*i as usize] = v;
    }
    scratch.last().cloned().unwrap_or(0.0)
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

fn pixel_g(ops: &[(u32, Op)], scratch: &mut Vec<Grad>, x: f32, y: f32) -> Grad {
    scratch.resize(ops.len(), Grad::from(0.0));
    for (i, op) in ops {
        let v = match op {
            Op::VarX => Grad::new(x, 1.0, 0.0, 0.0),
            Op::VarY => Grad::new(y, 0.0, 1.0, 0.0),
            Op::Add(a, b) => scratch[*a as usize] + scratch[*b as usize],
            Op::Sub(a, b) => scratch[*a as usize] - scratch[*b as usize],
            Op::Mul(a, b) => scratch[*a as usize] * scratch[*b as usize],
            Op::Div(a, b) => scratch[*a as usize] / scratch[*b as usize],
            Op::Max(a, b) => normalize(scratch[*a as usize])
                .max(normalize(scratch[*b as usize])),
            Op::Min(a, b) => normalize(scratch[*a as usize])
                .min(normalize(scratch[*b as usize])),
            Op::Const(f) => Grad::from(*f),
            Op::Square(a) => {
                let v = scratch[*a as usize];
                v * v
            }
            Op::Neg(a) => -scratch[*a as usize],
            Op::Sqrt(a) => scratch[*a as usize].sqrt(),
        };
        scratch[*i as usize] = v;
    }
    scratch.last().cloned().unwrap_or(Grad::from(0.0))
}

fn interval_i(
    ops: &[(u32, Op)],
    scratch: &mut Vec<Interval>,
    x: Interval,
    y: Interval,
) -> Interval {
    scratch.resize(ops.len(), Interval::from(0.0));
    for (i, op) in ops {
        let v = match op {
            Op::VarX => x,
            Op::VarY => y,
            Op::Add(a, b) => scratch[*a as usize] + scratch[*b as usize],
            Op::Sub(a, b) => scratch[*a as usize] - scratch[*b as usize],
            Op::Mul(a, b) => scratch[*a as usize] * scratch[*b as usize],
            Op::Div(a, b) => scratch[*a as usize] / scratch[*b as usize],
            Op::Max(a, b) => {
                scratch[*a as usize].max_choice(scratch[*b as usize]).0
            }
            Op::Min(a, b) => {
                scratch[*a as usize].min_choice(scratch[*b as usize]).0
            }
            Op::Const(f) => Interval::from(*f),
            Op::Square(a) => {
                let v = scratch[*a as usize];
                v * v
            }
            Op::Neg(a) => -scratch[*a as usize],
            Op::Sqrt(a) => scratch[*a as usize].sqrt(),
        };
        scratch[*i as usize] = v;
    }
    scratch.last().cloned().unwrap_or(Interval::from(0.0))
}

fn shade(f: f32) -> [u8; 4] {
    use fidget::render::RenderMode;
    let [r, g, b] = fidget::render::SdfPixelRenderMode::pixel(f);
    [r, g, b, 255]
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
    for iy in (0..args.size).rev() {
        let y = ((iy as f32) / (args.size as f32) - 0.5) * 2.0;
        for ix in 0..args.size {
            let x = ((ix as f32) / (args.size as f32) - 0.5) * 2.0;
            let v = if args.normalize {
                let g = pixel_g(&ops, &mut scratch_g, x, y);
                g.v * 2.0 // condensed field lines
            } else {
                pixel_f(&ops, &mut scratch_f, x, y)
            };
            pixels.push(shade(v));
        }
    }

    if let Some(out) = &args.out {
        let raw_pixels: Vec<u8> = pixels.into_iter().flatten().collect();
        let img: ImageBuffer<Rgba<u8>, _> =
            ImageBuffer::from_raw(args.size, args.size, raw_pixels)
                .expect("failed to create image buffer");
        img.save(out).expect("failed to save image");
    }
    println!("Hello, world!");
}

const PROSPERO: &'static str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/models/prospero.txt"));
