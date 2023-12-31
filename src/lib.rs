mod cpu;
mod state;
mod utils;

#[no_mangle]
pub extern "C" fn say_hi() {
    println!("hello from rust!");
}
