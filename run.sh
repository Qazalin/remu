cd ~/code/tinygrad/tinygrad

if git status | grep 'not staged' > /dev/null 2>&1; then
    scp -P 44 t.py qazal@72.220.147.45:/home/qazal/tinygrad/t.py > /dev/null 2>&1
    ssh -t -p 44 qazal@72.220.147.45 <<EOF > /dev/null 2>&1
    cd /home/qazal/tinygrad
    source venv/bin/activate && HIP=1 NOOPT=1 python3 t.py
    exit
EOF
    scp -P 44 qazal@72.220.147.45:/home/qazal/tinygrad/compiled.s /tmp/compiled.s
    git add t.py
fi

cd ~/code/tinygrad/remu
cargo build --release > /dev/null 2>&1

echo "remu:"
HIP=1 HIPCPU=1 REMU_DEBUG=1 NOOPT=1 python3 ~/code/tinygrad/tinygrad/t.py

echo "tinygrd:"
METAL=1 NOOPT=1 python3 ~/code/tinygrad/tinygrad/t.py
