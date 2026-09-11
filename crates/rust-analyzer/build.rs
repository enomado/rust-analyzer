//! Construct version in the `commit-hash date channel` format

use std::{env, path::PathBuf, process::Command};

fn main() {
    set_rerun();
    set_commit_info();
    if option_env!("CFG_RELEASE").is_none() {
        println!("cargo:rustc-env=POKE_RA_DEVS=1");
    }
}

fn set_rerun() {
    println!("cargo:rerun-if-env-changed=CFG_RELEASE");

    // Ask git where HEAD is recorded instead of guessing at a `.git` directory.
    // A linked worktree has a `.git` FILE naming the real git dir, so the walk
    // below finds nothing there — and a build script with no `rerun-if-changed`
    // is never re-run, freezing the reported commit at whatever was checked out
    // the first time this ran. The version then lies for the life of the target
    // directory, which is worse than having no version at all.
    #[allow(clippy::disallowed_methods)]
    if let Ok(output) = Command::new("git")
        .args(["rev-parse", "--path-format=absolute", "--git-path", "HEAD"])
        .output()
        && output.status.success()
        && let Ok(path) = String::from_utf8(output.stdout)
    {
        let path = path.trim();
        if !path.is_empty() {
            println!("cargo:rerun-if-changed={path}");
            return;
        }
    }

    let mut manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").expect("`CARGO_MANIFEST_DIR` is always set by cargo."),
    );

    while manifest_dir.parent().is_some() {
        let head_ref = manifest_dir.join(".git/HEAD");
        if head_ref.exists() {
            println!("cargo:rerun-if-changed={}", head_ref.display());
            return;
        }

        manifest_dir.pop();
    }

    println!("cargo:warning=Could not find `.git/HEAD` from manifest dir!");
}

fn set_commit_info() {
    #[allow(clippy::disallowed_methods)]
    let output = match Command::new("git")
        .arg("log")
        .arg("-1")
        .arg("--date=short")
        .arg("--format=%H %h %cd")
        .output()
    {
        Ok(output) if output.status.success() => output,
        _ => return,
    };
    let stdout = String::from_utf8(output.stdout).unwrap();
    let mut parts = stdout.split_whitespace();
    let mut next = || parts.next().unwrap();
    println!("cargo:rustc-env=RA_COMMIT_HASH={}", next());
    println!("cargo:rustc-env=RA_COMMIT_SHORT_HASH={}", next());
    println!("cargo:rustc-env=RA_COMMIT_DATE={}", next())
}
