//! Optional dependencies that `cargo metadata` resolves but cargo never builds.
//!
//! A feature may forward a feature to an optional dependency *weakly*, `dep?/feat`: "if `dep` is
//! enabled by something else, also enable its `feat`". Such a reference alone does not enable
//! `dep`, and `cargo build`/`cargo tree` leave it out. `cargo metadata` still lists an edge to
//! it in `resolve.nodes[].deps`. With nalgebra 0.34, whose default `std` feature carries
//! `glam014?/std` … `glam032?/std`, that is 18 versions of glam no build ever compiles; every
//! one of them would become a crate in the crate graph, with its def map, item trees and trait
//! impls (which the trait solver of every crate depending on nalgebra then gets to see).
//!
//! [`drop_unbuilt_optional_deps`] removes those edges, and then the packages no longer reachable
//! from a workspace member. The decision is made only from what the parent's manifest and its
//! enabled feature set say, never from package names: an edge to an optional dependency is kept
//! when an enabled feature of the parent enables it — the implicit feature named after it, a
//! `dep:name` entry or a strong `name/feat` entry. Whenever the edge cannot be matched to a
//! manifest dependency, it is kept.

use cargo_metadata::{FeatureName, Metadata, Node, NodeDep, Package, PackageId};
use rustc_hash::{FxHashMap, FxHashSet};

/// What [`drop_unbuilt_optional_deps`] removed, for the log.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DroppedUnbuilt {
    pub(crate) edges: usize,
    pub(crate) packages: usize,
}

/// Remove dependency edges cargo would not build, then packages no workspace member reaches.
///
/// Metadata without `resolve` (`--no-deps`) is left untouched.
pub(crate) fn drop_unbuilt_optional_deps(meta: &mut Metadata) -> DroppedUnbuilt {
    let Some(resolve) = meta.resolve.as_mut() else {
        return DroppedUnbuilt::default();
    };
    let packages: FxHashMap<&PackageId, &Package> =
        meta.packages.iter().map(|pkg| (&pkg.id, pkg)).collect();

    let mut dropped = DroppedUnbuilt::default();
    for node in &mut resolve.nodes {
        let parent = packages[&node.id];
        let enabled = enabled_dependency_names(parent, &node.features);
        let before = node.deps.len();
        node.deps.retain(|dep| is_built(parent, packages[&dep.pkg], dep, &enabled));
        dropped.edges += before - node.deps.len();
        // `dependencies` is the older, name-less list of the same edges.
        let kept: FxHashSet<&PackageId> = node.deps.iter().map(|dep| &dep.pkg).collect();
        node.dependencies.retain(|id| kept.contains(id));
    }

    // Packages that only hung off dropped edges are reachable from no member any more.
    let nodes: FxHashMap<&PackageId, &Node> =
        resolve.nodes.iter().map(|node| (&node.id, node)).collect();
    let mut reachable: FxHashSet<PackageId> = FxHashSet::default();
    let mut stack: Vec<&PackageId> = meta.workspace_members.iter().collect();
    while let Some(id) = stack.pop() {
        if !reachable.insert(id.clone()) {
            continue;
        }
        if let Some(node) = nodes.get(id) {
            stack.extend(node.deps.iter().map(|dep| &dep.pkg));
        }
    }
    // A package without a resolve node is not ours to judge: keep it.
    let has_node: FxHashSet<PackageId> = nodes.keys().map(|&id| id.clone()).collect();
    resolve.nodes.retain(|node| reachable.contains(&node.id));
    let before = meta.packages.len();
    meta.packages.retain(|pkg| reachable.contains(&pkg.id) || !has_node.contains(&pkg.id));
    dropped.packages = before - meta.packages.len();
    dropped
}

/// Names (rename if renamed, else package name) of the parent's optional dependencies that its
/// enabled features turn on, plus every enabled feature name itself.
///
/// `features` (the resolve node's) is cargo's closure of enabled features. An optional dependency
/// used without `dep:` anywhere has an implicit feature of its own name, which shows up there;
/// `dep:name` and strong `name/feat` turn the dependency on from inside another feature; weak
/// `name?/feat` does not.
fn enabled_dependency_names<'a>(
    parent: &'a Package,
    features: &'a [FeatureName],
) -> FxHashSet<&'a str> {
    let mut names: FxHashSet<&str> = features.iter().map(|feat| feat.as_str()).collect();
    for feat in features {
        let Some(entries) = parent.features.get(feat.as_str()) else { continue };
        for entry in entries {
            if let Some(name) = entry.strip_prefix("dep:") {
                names.insert(name);
            } else if let Some((name, _)) = entry.split_once('/')
                && !name.ends_with('?')
            {
                names.insert(name);
            }
        }
    }
    names
}

/// Whether cargo builds the edge `parent → child`.
///
/// The edge is matched to the parent's manifest dependencies by package name, version
/// requirement and kind (and, for a renamed dependency, by the extern name `dep.name`, which is
/// the rename with `-` turned into `_`). It is built when any match is not optional or is
/// enabled; with no match at all it is kept.
///
/// The kind matters: serde_with 3 has `schemars_0_9` both as an optional normal dependency
/// (only `schemars_0_9?/std` names it) and as a required dev-dependency. Dev-dependencies of a
/// non-member are not resolved, so the edge in the metadata is the normal one, and the dev
/// entry must not vouch for it.
fn is_built(parent: &Package, child: &Package, dep: &NodeDep, enabled: &FxHashSet<&str>) -> bool {
    let mut matched = false;
    for manifest_dep in &parent.dependencies {
        if manifest_dep.name != child.name.as_str() || !manifest_dep.req.matches(&child.version) {
            continue;
        }
        // An edge from an older cargo carries no kinds; then any kind matches.
        if !dep.dep_kinds.is_empty()
            && !dep.dep_kinds.iter().any(|info| info.kind == manifest_dep.kind)
        {
            continue;
        }
        if let Some(rename) = &manifest_dep.rename
            && rename.replace('-', "_") != dep.name
        {
            continue;
        }
        matched = true;
        let dep_name = manifest_dep.rename.as_deref().unwrap_or(&manifest_dep.name);
        if !manifest_dep.optional || enabled.contains(dep_name) {
            return true;
        }
    }
    !matched
}
