#!/bin/sh
BLOG=~/Web/blog/2025-05-15-gradients
cargo run --release -- -i models/circle.txt -o $BLOG/circle_interval@2x.png --size=768 --fancy
cargo run --release -- -i models/circle.txt -o $BLOG/circle@2x.png --size=768
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello@2x.png --size=768
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_interval@2x.png --size=768 --fancy
cargo run --release -- -i models/circle.txt -o $BLOG/circle_pseudo@2x.png --size=768 --fancy --pseudo

# These need to be post-processed in gimp
cargo run --release -- -i models/circle_pos.txt -o out.png --size=512 --fancy
./target/release/prospero-grad -i models/circle_pos.txt -o out_pseudo.png --size=512 --fancy --pseudo

# Same
cargo run --release -- -i models/rectangle_rot_circle.txt -o rot.png --size=512 --fancy
cargo run --release -- -i models/rectangle_rot_circle.txt -o rot_pseudo.png --size=512 --fancy --pseudo;

cargo run --release -- -i models/hello_world.txt --size 768 --save-grad -o $BLOG/hello_grad.png
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_norm_bad@2x.png --size=768 --bad-normalize
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_norm_good@2x.png --size=768 --normalize
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_pseudo@2x.png --size=768 --normalize --pseudo --fancy

# Need to be post-processed in Python
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_len_interval.png --size=512 --fancy --tape-len
cargo run --release -- -i models/hello_world.txt -o $BLOG/hello_len_pseudo.png --size=512 --normalize --pseudo --fancy --tape-len

cd ~/Web && make -j8
