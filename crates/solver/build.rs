fn main() {
    #[cfg(target_os = "macos")]
    {
        if cfg!(feature = "accelerate") {
            println!("cargo:rustc-link-lib=framework=Accelerate");
        }
    }
}
