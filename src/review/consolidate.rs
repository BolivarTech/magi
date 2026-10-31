// Author: Julian Bolivar
// Version: 0.24.3
// Date: 2026-10-22

//! Consolidation of the batch reports of one review into a single report (REQ-RV12), and the
//! agreement weight applied after the gate (REQ-RV11).
//!
//! Consensus, deduplication and the highest severity all come from magi-core's
//! [`ConsensusEngine`]; this module only builds its input (one synthetic output per mage) and
//! maps every reported finding back to the group magi-core put it in, so no title, detail or
//! category is lost when findings merge.

use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet, HashMap};

use magi_core::consensus::{ConsensusEngine, ConsensusResult, DedupFinding};
use magi_core::finding_id::generate_finding_id;
use magi_core::schema::{AgentName, AgentOutput, Category, Finding, Severity, Verdict};

use super::error::ReviewError;
use crate::redact::redact_foreign_text;

/// Run-log target of the consolidation WARN entries.
const LOG_TARGET: &str = "magi_rs::review";

/// The category every located finding carries into the consensus engine, so its key is
/// file + line alone: a mage-reported category is unverified and must not split one defect
/// into two single-mage findings.
const NEUTRAL_CATEGORY: Category = Category::Other;

/// What one batch's `consult` returned: the mages that answered and their outputs.
#[derive(Debug, Clone)]
pub struct BatchOutcome {
    /// Zero-based batch index.
    pub batch: usize,
    /// The outputs of the mages that answered, as the batch report carried them (raw).
    pub agents: Vec<AgentOutput>,
    /// The mages that answered in this batch (the `agent` of each report entry).
    pub answered: BTreeSet<AgentName>,
}

/// One finding of the consolidated report.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ConsolidatedFinding {
    /// The reported file, if any.
    pub file: Option<String>,
    /// The reported 1-based line, if any.
    pub line: Option<u32>,
    /// The severity after the agreement weight ([`apply_agreement`]).
    pub severity: Severity,
    /// The highest severity the trio reported; what the gate and the critical rule read.
    pub reported_severity: Severity,
    /// Every title merged into this finding, redacted, in order of appearance.
    pub titles: Vec<String>,
    /// Every detail merged into this finding, redacted, in order of appearance.
    pub details: Vec<String>,
    /// Every category the mages reported, in order of appearance.
    pub categories: Vec<Category>,
    /// The mages that reported it, each once.
    pub sources: Vec<AgentName>,
    /// Marks set by consolidation and by the agreement weight.
    pub marks: Vec<FindingMark>,
    /// The batches where it appeared, ascending; 0-based, serialized 1-based.
    #[serde(serialize_with = "super::report::serialize_batches")]
    pub batches: Vec<usize>,
}

/// A mark on a consolidated finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingMark {
    /// Only one mage reported it while two or more answered; one severity level lower.
    SingleMage,
    /// Only one mage answered where it appeared; there was no disagreement to weigh.
    PartialTrio,
    /// An unlocated finding seen by one mage in a review that spent [`MAX_UNLOCATED_PROBES`].
    /// Once the probes ran out, any later unlocated finding may have belonged to its group
    /// without being asked, so a single source no longer proves disagreement: it is not
    /// lowered. Set on every such finding, including one grouped before the budget ran out;
    /// [`Consolidated::ungrouped_unlocated`] counts only the findings whose own lookup was
    /// refused.
    Ungrouped,
}

/// The consolidated review.
#[derive(Debug, Clone)]
pub struct Consolidated {
    /// One synthetic output per mage, in order of first appearance.
    pub agents: Vec<AgentOutput>,
    /// The consensus magi-core computed over [`Consolidated::agents`].
    pub consensus: ConsensusResult,
    /// The consolidated findings, in the consensus order (highest severity first), followed
    /// by any group formed outside it.
    pub findings: Vec<ConsolidatedFinding>,
    /// Which mages answered in each batch.
    pub answered: BTreeMap<usize, BTreeSet<AgentName>>,
    /// Unlocated findings that formed their own group because the probe budget was spent
    /// (declared, never dropped).
    pub ungrouped_unlocated: usize,
}

/// Probes (`determine` on two one-finding outputs) one review may spend grouping unlocated
/// findings. A probe measured ~7.3 µs in release (~64 µs in debug) on this implementation, so
/// the whole budget costs about 0.7 s. A starting point, not a measurement of need.
pub const MAX_UNLOCATED_PROBES: usize = 100_000;

/// Merges the batch reports of one review into one report (REQ-RV12).
///
/// Per mage, one output is built from a clone of its worst-verdict batch output (`reject` >
/// `conditional` > `approve`; the first such batch wins a tie) with the lowest confidence
/// across batches, every batch's summary and reasoning labelled `[batch k]` (1-based) in
/// order, and the union of its findings. Every located finding (non-empty `file` and
/// `line > 0`, the same split magi-core's key makes) carries the neutral category, so two
/// mages that place a defect on the same line are one finding. The outputs go to
/// [`ConsensusEngine::determine`], which gives the consensus, the dedup and the highest
/// severity. Model text (titles, details, summaries, reasoning, recommendations) is redacted
/// on the way in.
///
/// The `file` field is not redacted here, on purpose: the input is [`super::validate`]'s output,
/// which has already replaced every kept finding's `file` with a reviewed path (or a rename
/// target) or `None`, so no model-written path reaches this point, and that path is the key the
/// gate uses to find the code it shows. Redacting a key could break the lookup. A caller that
/// skips validation must not use this function. The report redacts `file` on the way out with
/// the URL-credential and RV22 detectors (`report::redacted`), and the gate refuses any state
/// that holds a secret.
///
/// Each reported finding is then mapped to its group without a local key: a located one by
/// [`generate_finding_id`] with the neutral category, an unlocated one by asking magi-core
/// whether its title and a group's title are one finding (a probe), with identical titles
/// decided without a call and each pair probed once. Every title, detail and original
/// category is kept, in order of appearance.
///
/// # Complexity
///
/// O(n) to build the outputs and map located findings (n reported findings), plus magi-core's
/// own O(n²) dedup, plus at most O(u·g) probes for u unlocated findings over g unlocated groups.
/// A title already placed (a group's or an earlier member's) is found in O(1) without a probe,
/// and a new title's scan stops at the first group that accepts it, each step a probe, so the
/// grouping work is O(u + probes). The probes are per review, not per batch (u and g grow with
/// the batches), and bounded by [`MAX_UNLOCATED_PROBES`]: past it an unlocated finding whose
/// title was not placed before forms its own group at O(1), is counted in [`Consolidated::ungrouped_unlocated`], and
/// every single-source unlocated finding is marked [`FindingMark::Ungrouped`].
///
/// # Arguments
///
/// * `batches` - The outcome of every batch that ran, in batch order.
///
/// # Returns
///
/// The consolidated review.
///
/// # Errors
///
/// [`ReviewError::Consolidation`] when magi-core refuses the consensus: no batch, or fewer
/// mages across the review than its `min_agents` (2).
// MAGI FIX (S6 loop 6): the unredacted `file` is a contract on the input, now stated.
pub fn consolidate(batches: &[BatchOutcome]) -> Result<Consolidated, ReviewError> {
    consolidate_capped(batches, MAX_UNLOCATED_PROBES)
}

/// [`consolidate`] with an explicit probe budget.
#[cfg(test)]
pub(crate) fn consolidate_with_probe_cap(
    batches: &[BatchOutcome],
    cap: usize,
) -> Result<Consolidated, ReviewError> {
    consolidate_capped(batches, cap)
}

/// One reported finding, redacted, with where it came from.
struct Reported {
    /// The batch it came from.
    batch: usize,
    /// The mage that reported it.
    agent: AgentName,
    /// The finding with redacted text and its original category.
    finding: Finding,
}

/// The per-mage accumulator behind one synthetic output.
struct MageAcc {
    /// A clone of the mage's worst-verdict output; supplies every field not set below.
    base: AgentOutput,
    /// The lowest confidence across batches.
    confidence: f64,
    /// `[batch k] summary`, in batch order.
    summaries: Vec<String>,
    /// `[batch k] reasoning`, in batch order.
    reasoning: Vec<String>,
    /// The union of the mage's findings, with the neutral category when located.
    findings: Vec<Finding>,
}

/// A group of reported findings that are one consolidated finding.
struct Group {
    /// The magi-core group it stands for, if any.
    dedup: Option<usize>,
    /// The stable id, for a located group.
    id: Option<String>,
    /// The representative title an unlocated finding is compared against.
    rep: String,
    /// Indices into the reported findings, in order of appearance.
    members: Vec<usize>,
}

/// The body of [`consolidate`], with the probe budget as a parameter.
fn consolidate_capped(batches: &[BatchOutcome], cap: usize) -> Result<Consolidated, ReviewError> {
    let (agents, reported) = synthesize(batches);
    let engine = ConsensusEngine::default();
    let consensus = engine
        .determine(&agents)
        .map_err(|e| ReviewError::Consolidation(redacted(&e.to_string())))?;
    let probe = Prober::new(&engine, agents.first(), cap);
    let mut groups: Vec<Group> = consensus
        .findings
        .iter()
        .enumerate()
        .map(|(i, d)| Group {
            dedup: Some(i),
            id: d.id.clone(),
            rep: d.title.clone(),
            members: Vec::new(),
        })
        .collect();
    let mut by_id: HashMap<String, usize> = groups
        .iter()
        .enumerate()
        .filter_map(|(i, g)| g.id.clone().map(|id| (id, i)))
        .collect();
    let mut unlocated = Unlocated::new(&groups);
    let mut ungrouped_unlocated = 0usize;
    for (k, r) in reported.iter().enumerate() {
        let located = located(&r.finding);
        let slot = match located {
            Some((file, line)) => {
                let id = generate_finding_id(file, line, NEUTRAL_CATEGORY);
                by_id.get(&id).copied().ok_or(Some(id))
            }
            None => unlocated
                .slot(&groups, &r.finding.title, &probe)
                .ok_or(None),
        };
        match slot {
            Ok(i) => {
                if let Some(g) = groups.get_mut(i) {
                    g.members.push(k);
                }
                if located.is_none() {
                    unlocated.place(&r.finding.title, i, false);
                }
            }
            Err(id) => {
                if located.is_none() && probe.refused.get() {
                    ungrouped_unlocated += 1;
                } else {
                    tracing::event!(
                        target: LOG_TARGET,
                        tracing::Level::WARN,
                        "a finding no consolidated group claimed was kept as its own finding"
                    );
                }
                match &id {
                    Some(id) => {
                        by_id.insert(id.clone(), groups.len());
                    }
                    None => unlocated.place(&r.finding.title, groups.len(), true),
                }
                groups.push(Group {
                    dedup: None,
                    id,
                    rep: r.finding.title.clone(),
                    members: vec![k],
                });
            }
        }
    }
    let exhausted = probe.exhausted.get();
    let findings = groups
        .iter()
        .filter(|g| !g.members.is_empty())
        .map(|g| finding_of(g, &reported, &consensus.findings, exhausted))
        .collect();
    let answered = batches
        .iter()
        .map(|b| (b.batch, b.answered.clone()))
        .collect();
    Ok(Consolidated {
        agents,
        consensus,
        findings,
        answered,
        ungrouped_unlocated,
    })
}

/// The unlocated groups of a review (the groups without an id), indexed by title.
struct Unlocated {
    /// Indices into the review's groups of the groups without an id, in creation order.
    slots: Vec<usize>,
    /// Every unlocated title already placed, to the group it went to (the first placement wins).
    by_title: HashMap<String, usize>,
}

impl Unlocated {
    /// The index over the groups magi-core formed: each one without an id, under its title.
    fn new(groups: &[Group]) -> Self {
        let mut index = Self {
            slots: Vec::new(),
            by_title: HashMap::new(),
        };
        for (i, g) in groups.iter().enumerate().filter(|(_, g)| g.id.is_none()) {
            index.place(&g.rep, i, true);
        }
        index
    }

    /// Records that `title` went to group `slot`; `new_group` when that group was just made.
    fn place(&mut self, title: &str, slot: usize, new_group: bool) {
        if new_group {
            self.slots.push(slot);
        }
        self.by_title.entry(title.to_string()).or_insert(slot);
    }

    /// The group (index into `groups`) an unlocated title belongs to; resets `probe.refused`
    /// so the caller can tell a refused probe from no match.
    ///
    /// A title already placed (a group's representative or an earlier member) decides without
    /// a probe; otherwise the first group, in creation order, that `probe` accepts. `None` when
    /// none does.
    ///
    /// # Complexity
    ///
    /// O(1) for a title already placed. A new title asks the groups in order until one accepts
    /// it, and each of those asks is a probe (a new title has no memoized pair), so the scans of a
    /// review cost at most its probes plus one per lookup. Once the budget is spent a new title
    /// asks no group: it is refused at once (O(1)), so the cap bounds the work as well as the
    /// `determine` calls.
    // MAGI FIX (S6 loop 5): the exact-title scan is a map and a spent budget asks no group.
    fn slot(&self, groups: &[Group], title: &str, probe: &Prober<'_>) -> Option<usize> {
        probe.refused.set(false);
        if let Some(&i) = self.by_title.get(title) {
            return Some(i);
        }
        if probe.left.get() == 0 {
            let refused = !self.slots.is_empty();
            probe.refused.set(refused);
            if refused {
                probe.exhausted.set(true);
            }
            return None;
        }
        self.slots
            .iter()
            .copied()
            .find(|&i| groups.get(i).is_some_and(|g| probe.same(title, &g.rep)))
    }
}

/// Builds one synthetic output per mage and the list of reported findings.
fn synthesize(batches: &[BatchOutcome]) -> (Vec<AgentOutput>, Vec<Reported>) {
    let mut mages: Vec<(AgentName, MageAcc)> = Vec::new();
    let mut reported = Vec::new();
    for b in batches {
        for a in &b.agents {
            let label = b.batch + 1;
            let mut neutral = Vec::with_capacity(a.findings.len());
            for f in &a.findings {
                let mut clean = f.clone();
                clean.title = redacted(&f.title);
                clean.detail = redacted(&f.detail);
                let mut keyed = clean.clone();
                if located(&keyed).is_some() {
                    keyed.category = NEUTRAL_CATEGORY;
                }
                neutral.push(keyed);
                reported.push(Reported {
                    batch: b.batch,
                    agent: a.agent,
                    finding: clean,
                });
            }
            let summary = format!("[batch {label}] {}", redacted(&a.summary));
            let reasoning = format!("[batch {label}] {}", redacted(&a.reasoning));
            match mages.iter_mut().find(|(n, _)| *n == a.agent) {
                Some((_, acc)) => {
                    if rank(a.verdict) > rank(acc.base.verdict) {
                        acc.base = a.clone();
                    }
                    acc.confidence = acc.confidence.min(a.confidence);
                    acc.summaries.push(summary);
                    acc.reasoning.push(reasoning);
                    acc.findings.extend(neutral);
                }
                None => mages.push((
                    a.agent,
                    MageAcc {
                        base: a.clone(),
                        confidence: a.confidence,
                        summaries: vec![summary],
                        reasoning: vec![reasoning],
                        findings: neutral,
                    },
                )),
            }
        }
    }
    let agents = mages
        .into_iter()
        .map(|(_, acc)| {
            let mut out = acc.base;
            out.confidence = acc.confidence;
            out.summary = acc.summaries.join("\n");
            out.reasoning = acc.reasoning.join("\n");
            out.recommendation = redacted(&out.recommendation);
            out.findings = acc.findings;
            out
        })
        .collect();
    (agents, reported)
}

/// Builds the consolidated finding of one group.
fn finding_of(
    g: &Group,
    reported: &[Reported],
    dedup: &[DedupFinding],
    exhausted: bool,
) -> ConsolidatedFinding {
    let members: Vec<&Reported> = g.members.iter().filter_map(|&k| reported.get(k)).collect();
    let present: BTreeSet<AgentName> = members.iter().map(|r| r.agent).collect();
    let from_core = g.dedup.and_then(|i| dedup.get(i));
    // magi-core's sources first, in its order; a group the probe cap split keeps only the mages
    // whose findings it still holds, so no source counts in two groups. Then every other mage
    // whose finding the group holds, in order of appearance.
    // MAGI FIX (S6 loop 9, C1-m1): past magi-core's findings cap, a mage's finding it dropped
    // joins another mage's group at the same location; magi-core's list does not name that mage,
    // and reading only that list made a finding of two mages `single_mage`.
    let mut sources: Vec<AgentName> = from_core
        .map(|d| {
            d.sources
                .iter()
                .copied()
                .filter(|a| present.contains(a))
                .collect()
        })
        .unwrap_or_default();
    for r in &members {
        if !sources.contains(&r.agent) {
            sources.push(r.agent);
        }
    }
    let reported_severity = members
        .iter()
        .map(|r| r.finding.severity)
        .max()
        .unwrap_or(Severity::Info);
    let first = members.first().map(|r| &r.finding);
    let (file, line) = match from_core {
        Some(d) => (d.file.clone(), d.line),
        None => (
            first.and_then(|f| f.file.clone()),
            first.and_then(|f| f.line),
        ),
    };
    let unlocated = first.is_some_and(|f| located(f).is_none());
    let marks = if exhausted && unlocated && sources.len() == 1 {
        vec![FindingMark::Ungrouped]
    } else {
        Vec::new()
    };
    let mut batches: Vec<usize> = members.iter().map(|r| r.batch).collect();
    batches.sort_unstable();
    batches.dedup();
    ConsolidatedFinding {
        file,
        line,
        severity: reported_severity,
        reported_severity,
        titles: members.iter().map(|r| r.finding.title.clone()).collect(),
        details: members.iter().map(|r| r.finding.detail.clone()).collect(),
        categories: members.iter().map(|r| r.finding.category).collect(),
        sources,
        marks,
        batches,
    }
}

/// Asks magi-core whether two unlocated titles are one finding, within a budget.
struct Prober<'a> {
    /// The engine whose private title key decides.
    engine: &'a ConsensusEngine,
    /// An empty output both probe outputs are cloned from; `None` with no mage at all.
    template: Option<AgentOutput>,
    /// Probes left.
    left: Cell<usize>,
    /// Pairs already probed.
    memo: RefCell<HashMap<(String, String), bool>>,
    /// Whether the current lookup was refused a probe.
    refused: Cell<bool>,
    /// Whether any lookup of the review was refused a probe.
    exhausted: Cell<bool>,
    /// Calls of [`Self::same`], so a test can see what a lookup cost.
    #[cfg(test)]
    same_calls: Cell<usize>,
}

impl<'a> Prober<'a> {
    /// A prober over `engine` with `cap` probes, cloning its outputs from `any`.
    fn new(engine: &'a ConsensusEngine, any: Option<&AgentOutput>, cap: usize) -> Self {
        let template = any.map(|a| {
            let mut t = a.clone();
            t.summary.clear();
            t.reasoning.clear();
            t.recommendation.clear();
            t.findings.clear();
            t
        });
        Self {
            engine,
            template,
            left: Cell::new(cap),
            memo: RefCell::new(HashMap::new()),
            refused: Cell::new(false),
            exhausted: Cell::new(false),
            #[cfg(test)]
            same_calls: Cell::new(0),
        }
    }

    /// Whether magi-core keys `title` and `rep` as one finding; `false` once the budget is
    /// spent (recorded in `refused` and `exhausted`).
    fn same(&self, title: &str, rep: &str) -> bool {
        #[cfg(test)]
        self.same_calls.set(self.same_calls.get() + 1);
        let key = (title.to_string(), rep.to_string());
        if let Some(&known) = self.memo.borrow().get(&key) {
            return known;
        }
        let left = self.left.get();
        if left == 0 {
            self.refused.set(true);
            self.exhausted.set(true);
            return false;
        }
        self.left.set(left - 1);
        let answer = self.probe(title, rep);
        self.memo.borrow_mut().insert(key, answer);
        answer
    }

    /// One probe: `determine` over two one-finding outputs gives one finding iff one key.
    fn probe(&self, title: &str, rep: &str) -> bool {
        let Some(template) = &self.template else {
            return false;
        };
        let one = |agent: AgentName, t: &str| {
            let mut o = template.clone();
            o.agent = agent;
            o.findings = vec![Finding::new(Severity::Info, t, "")];
            o
        };
        // Two distinct mages, because `determine` refuses fewer than its `min_agents` (2 in
        // magi-core 4.2.0, pinned by `magi_core_still_refuses_a_one_mage_consensus`). Were that
        // minimum to rise, `determine` would fail, every probe would answer "not the same" and
        // each unlocated title would keep its own group: no grouping, but nothing lost.
        let pair = [
            one(AgentName::Melchior, title),
            one(AgentName::Balthasar, rep),
        ];
        self.engine
            .determine(&pair)
            .is_ok_and(|c| c.findings.len() == 1)
    }
}

/// The location magi-core keys by id: a non-empty file and a positive line.
fn located(f: &Finding) -> Option<(&str, u32)> {
    match (f.file.as_deref(), f.line) {
        (Some(file), Some(line)) if !file.is_empty() && line > 0 => Some((file, line)),
        _ => None,
    }
}

/// How bad a verdict is: `approve` < `conditional` < `reject`.
fn rank(v: Verdict) -> u8 {
    match v {
        Verdict::Approve => 0,
        Verdict::Conditional => 1,
        Verdict::Reject => 2,
    }
}

/// Model text with credentials redacted, as `consult` writes it.
fn redacted(raw: &str) -> String {
    redact_foreign_text(raw).as_str().to_string()
}

/// Applies the agreement weight (REQ-RV11); runs after the gate.
///
/// Who answered is the union of `answered` over the batches where the finding appeared. With
/// one mage or none there is no disagreement to weigh: the finding keeps its severity and is
/// marked [`FindingMark::PartialTrio`]. With two or more and a single source, it drops one
/// level (`critical` → `warning` → `info`; `info` stays) and is marked
/// [`FindingMark::SingleMage`]. A finding marked [`FindingMark::Ungrouped`] is left as it is:
/// its agreement was never measured. Nothing is ever removed.
///
/// # Arguments
///
/// * `findings` - The consolidated findings, changed in place.
/// * `answered` - Which mages answered in each batch ([`Consolidated::answered`]).
pub fn apply_agreement(
    findings: &mut [ConsolidatedFinding],
    answered: &BTreeMap<usize, BTreeSet<AgentName>>,
) {
    for f in findings.iter_mut() {
        if f.marks.contains(&FindingMark::Ungrouped) {
            continue;
        }
        let responders: BTreeSet<AgentName> = f
            .batches
            .iter()
            .filter_map(|b| answered.get(b))
            .flatten()
            .copied()
            .collect();
        if responders.len() < 2 {
            f.marks.push(FindingMark::PartialTrio);
        } else if f.sources.len() == 1 {
            f.severity = lower(f.severity);
            f.marks.push(FindingMark::SingleMage);
        }
    }
}

/// One severity level lower; `info` stays `info`.
fn lower(s: Severity) -> Severity {
    match s {
        Severity::Critical => Severity::Warning,
        Severity::Warning | Severity::Info => Severity::Info,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::review::test_support::{out, real_report};

    /// `n` unlocated groups titled `r0`, `r1`, ...
    fn unlocated_groups(n: usize) -> Vec<Group> {
        (0..n)
            .map(|i| Group {
                dedup: None,
                id: None,
                rep: format!("r{i}"),
                members: Vec::new(),
            })
            .collect()
    }

    #[test]
    fn a_spent_probe_budget_places_an_unlocated_title_without_asking_any_group() {
        // [MAGI S6 loop 5, C1-c2] the cap bounds the work, not only the `determine` calls: once
        // it is spent, a new title is its own group at no cost per group.
        let groups = unlocated_groups(1_000);
        let engine = ConsensusEngine::default();
        let probe = Prober::new(&engine, None, 0);
        let index = Unlocated::new(&groups);
        assert_eq!(index.slot(&groups, "new title", &probe), None);
        assert!(probe.refused.get() && probe.exhausted.get());
        assert_eq!(probe.same_calls.get(), 0);
    }

    #[test]
    fn a_title_already_placed_finds_its_group_without_asking_any_group() {
        let groups = unlocated_groups(1_000);
        let engine = ConsensusEngine::default();
        let probe = Prober::new(&engine, None, 10);
        let mut index = Unlocated::new(&groups);
        index.place("seen", 999, false);
        assert_eq!(index.slot(&groups, "seen", &probe), Some(999));
        assert_eq!(index.slot(&groups, "r500", &probe), Some(500));
        assert!(!probe.refused.get());
        assert_eq!(probe.same_calls.get(), 0);
    }
    use magi_core::schema::Verdict;

    fn test_finding() -> ConsolidatedFinding {
        ConsolidatedFinding {
            file: Some("a.rs".into()),
            line: Some(1),
            severity: Severity::Info,
            reported_severity: Severity::Info,
            titles: vec!["t".into()],
            details: vec!["d".into()],
            categories: vec![Category::Other],
            sources: vec![AgentName::Melchior],
            marks: vec![],
            batches: vec![0],
        }
    }
    fn batch(i: usize, agents: Vec<AgentOutput>) -> BatchOutcome {
        let answered = agents.iter().map(|a| a.agent).collect();
        BatchOutcome {
            batch: i,
            agents,
            answered,
        }
    }

    #[test]
    fn a_mage_takes_its_worst_verdict_across_batches() {
        let c = consolidate(&[
            batch(
                0,
                vec![
                    out("melchior", "approve", 0.9, serde_json::json!([])),
                    out("balthasar", "approve", 0.9, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
            batch(
                1,
                vec![
                    out("melchior", "reject", 0.7, serde_json::json!([])),
                    out("balthasar", "approve", 0.8, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
        ])
        .expect("consolidates");
        let m = c
            .agents
            .iter()
            .find(|a| a.agent == AgentName::Melchior)
            .expect("melchior");
        assert_eq!(m.verdict, Verdict::Reject);
        assert!((m.confidence - 0.7).abs() < 1e-9);
        assert_eq!(
            c.consensus.votes.get(&AgentName::Melchior),
            Some(&Verdict::Reject)
        );
    }

    #[test]
    fn the_same_finding_in_two_batches_appears_once_at_the_higher_severity() {
        let f = |sev: &str| serde_json::json!([{"severity": sev, "title":"x","detail":"d","file":"a.rs","line":3}]);
        // [RED-DRYRUN 2026-10-17, D14] A second mage per batch: alone, Melchior is one agent and
        // ConsensusEngine::default() (min_agents = 2) returns InsufficientAgents, which T2.9 relies on as a
        // consolidation failure.
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let c = consolidate(&[
            batch(
                0,
                vec![out("melchior", "approve", 0.9, f("warning")), quiet()],
            ),
            batch(
                1,
                vec![out("melchior", "approve", 0.9, f("critical")), quiet()],
            ),
        ])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].reported_severity, Severity::Critical);
    }

    #[test]
    fn different_categories_on_the_same_line_are_one_finding_with_both_sources() {
        let c = consolidate(&[batch(0, vec![
            out("melchior", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"a","detail":"da","file":"src/a.rs","line":10,"category":"logic-error"}])),
            out("caspar", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"b","detail":"db","file":"src/a.rs","line":10,"category":"error-handling"}])),
        ])]).expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        let f = &c.findings[0];
        assert_eq!(f.sources.len(), 2);
        assert_eq!(f.titles, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(f.details, vec!["da".to_string(), "db".to_string()]);
        assert_eq!(
            f.categories,
            vec![Category::LogicError, Category::ErrorHandling]
        );
    }

    #[test]
    fn a_single_mage_finding_drops_one_level_and_a_shared_one_keeps_it() {
        let mut fs = vec![
            ConsolidatedFinding {
                severity: Severity::Critical,
                reported_severity: Severity::Critical,
                sources: vec![AgentName::Melchior],
                ..test_finding()
            },
            ConsolidatedFinding {
                severity: Severity::Warning,
                reported_severity: Severity::Warning,
                sources: vec![AgentName::Melchior, AgentName::Caspar],
                ..test_finding()
            },
            ConsolidatedFinding {
                severity: Severity::Info,
                reported_severity: Severity::Info,
                sources: vec![AgentName::Caspar],
                ..test_finding()
            },
        ];
        let all: BTreeSet<AgentName> =
            [AgentName::Melchior, AgentName::Balthasar, AgentName::Caspar]
                .into_iter()
                .collect();
        apply_agreement(&mut fs, &[(0, all)].into_iter().collect());
        assert_eq!(
            (fs[0].severity, fs[0].marks.clone()),
            (Severity::Warning, vec![FindingMark::SingleMage])
        );
        assert_eq!(
            (fs[1].severity, fs[1].marks.clone()),
            (Severity::Warning, vec![])
        );
        assert_eq!(
            (fs[2].severity, fs[2].marks.clone()),
            (Severity::Info, vec![FindingMark::SingleMage])
        );
    }

    #[test]
    fn when_only_one_mage_answered_there_is_no_disagreement_to_weigh() {
        let mut fs = vec![ConsolidatedFinding {
            severity: Severity::Critical,
            reported_severity: Severity::Critical,
            sources: vec![AgentName::Melchior],
            ..test_finding()
        }];
        apply_agreement(
            &mut fs,
            &[(0, [AgentName::Melchior].into_iter().collect())]
                .into_iter()
                .collect(),
        );
        assert_eq!(
            (fs[0].severity, fs[0].marks.clone()),
            (Severity::Critical, vec![FindingMark::PartialTrio])
        );
    }

    #[tokio::test]
    async fn consolidating_a_real_report_matches_its_own_consensus() {
        // PM-S-1: the AgentOutputs come from magi-core's orchestrator, not from a literal.
        let f = serde_json::json!([{"severity":"warning","title":"x","detail":"d","file":"a.rs","line":3}]);
        let report = real_report([f.clone(), f, serde_json::json!([])]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        assert_eq!(
            c.consensus.consensus_verdict,
            report.consensus.consensus_verdict
        );
        assert_eq!(c.consensus.votes, report.consensus.votes);
        assert_eq!(c.findings.len(), report.consensus.findings.len());
        assert_eq!(
            c.findings[0].sources.len(),
            report.consensus.findings[0].sources.len()
        );
    }

    #[tokio::test]
    async fn unlocated_findings_group_like_magi_core() {
        // [CP2 MS2 S2 loop 2] PM-S-1 for unlocated findings: same title, different categories → magi-core's key is the title alone.
        let m = serde_json::json!([{"severity":"warning","title":"Leaks  A Handle","detail":"d","category":"logic-error"}]);
        let c2 = serde_json::json!([{"severity":"info","title":"leaks  a handle","detail":"d2","category":"error-handling"}]);
        let report = real_report([m, serde_json::json!([]), c2]).await;
        let answered = report.agents.iter().map(|a| a.agent).collect();
        let c = consolidate(&[BatchOutcome {
            batch: 0,
            agents: report.agents.clone(),
            answered,
        }])
        .expect("consolidates");
        assert_eq!(c.findings.len(), report.consensus.findings.len());
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].sources.len(), 2);
    }

    #[test]
    fn a_finding_with_a_file_and_no_line_groups_by_title_like_magi_core() {
        // [CP2 MS2 S2a loop 7] load-bearing for the sondas: magi-core keys `file` + no line by title (consensus.rs:210).
        // Two mages, same title, different files, no line → one group, both sources. Goes on the magi-core bump checklist.
        let a = serde_json::json!([{"severity":"warning","title":"leaks a handle","detail":"da","file":"a.rs"}]);
        let b = serde_json::json!([{"severity":"warning","title":"leaks a handle","detail":"db","file":"b.rs"}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, a),
                out("caspar", "approve", 0.9, b),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(
            c.findings[0].sources,
            vec![AgentName::Melchior, AgentName::Caspar]
        );
    }

    #[test]
    fn past_the_probe_cap_unlocated_findings_stay_apart_and_are_counted() {
        // [CP2 MS2 S2a loop 8] the O(u·g) probe is bounded: past the cap no probe runs and nothing is dropped.
        let m = serde_json::json!([{"severity":"warning","title":"Leaks  A Handle","detail":"d"}]);
        let c2 =
            serde_json::json!([{"severity":"warning","title":"leaks  a handle","detail":"d2"}]);
        let b = [batch(
            0,
            vec![
                out("melchior", "approve", 0.9, m),
                out("caspar", "approve", 0.9, c2),
            ],
        )];
        let open = consolidate(&b).expect("consolidates");
        assert_eq!(
            (open.findings.len(), open.ungrouped_unlocated),
            (1, 0),
            "magi-core merges them"
        );
        let capped = consolidate_with_probe_cap(&b, 0).expect("consolidates");
        assert_eq!(
            capped.findings.len(),
            2,
            "no probe ran: the second title is its own group"
        );
        assert_eq!(capped.ungrouped_unlocated, 1);
        // [CP2 MS2 cierre (sin MAGI)] a fallback group's sources come from its own input findings, never from a group it left.
        assert_eq!(
            capped
                .findings
                .iter()
                .map(|f| f.sources.clone())
                .collect::<Vec<_>>(),
            vec![vec![AgentName::Melchior], vec![AgentName::Caspar]]
        );
        // [CP2 MS2 S2a loop 9] past the cap the agreement was never measured: marked, and no single_mage downgrade.
        assert!(
            capped
                .findings
                .iter()
                .all(|f| f.marks == vec![FindingMark::Ungrouped]),
            "{:?}",
            capped.findings
        );
        let mut fs = capped.findings.clone();
        apply_agreement(&mut fs, &capped.answered);
        assert!(fs.iter().all(
            |f| f.severity == Severity::Warning && !f.marks.contains(&FindingMark::SingleMage)
        ));
    }

    #[test]
    fn membership_is_decided_by_magi_core_not_by_a_local_key() {
        // [CP2 MS2 S2 loop 4] "\u{fb01}" (the "fi" ligature) folds to "fi" under NFKC: magi-core groups both titles; a naive
        // lowercase key would not. Both texts survive the merge (RV12, no loss).
        let a = serde_json::json!([{"severity":"warning","title":"Leaks a \u{fb01}le handle","detail":"da"}]);
        let b =
            serde_json::json!([{"severity":"warning","title":"leaks a file handle","detail":"db"}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, a),
                out("caspar", "approve", 0.9, b),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 1);
        assert_eq!(c.findings[0].titles.len(), 2);
        assert_eq!(
            c.findings[0].sources,
            vec![AgentName::Melchior, AgentName::Caspar]
        );
    }

    #[test]
    fn a_mage_repeating_a_finding_across_batches_is_one_source() {
        // [CP2 MS2 S2 loop 4] sources come from magi-core (E18): same mage, same key, two batches → one source → single_mage.
        let f = serde_json::json!([{"severity":"critical","title":"x","detail":"d","file":"a.rs","line":3}]);
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let mut c = consolidate(&[
            batch(0, vec![out("melchior", "approve", 0.9, f.clone()), quiet()]),
            batch(1, vec![out("melchior", "approve", 0.9, f), quiet()]),
        ])
        .expect("consolidates");
        assert_eq!(c.findings[0].sources, vec![AgentName::Melchior]);
        assert_eq!(c.findings[0].batches, vec![0, 1]);
        let answered = c.answered.clone();
        apply_agreement(&mut c.findings, &answered);
        assert_eq!(
            (c.findings[0].severity, c.findings[0].marks.clone()),
            (Severity::Warning, vec![FindingMark::SingleMage])
        );
    }

    #[test]
    fn a_finding_no_group_claims_is_never_dropped() {
        // [CP2 MS2 S2 loop 5] the fallback: the lookup says None, so consolidate gives the finding its own group (RV12, no loss).
        // [MAGI S6 loop 5, C1-c2] moved from `group_of` to `Unlocated::slot`, which replaced it.
        let mut groups = unlocated_groups(0);
        groups.push(Group {
            dedup: None,
            id: None,
            rep: "y".into(),
            members: Vec::new(),
        });
        let engine = ConsensusEngine::default();
        let probe = Prober::new(&engine, None, 10);
        let index = Unlocated::new(&groups);
        assert_eq!(index.slot(&groups, "x", &probe), None);
        assert_eq!(index.slot(&groups, "y", &probe), Some(0));
        assert_eq!(
            probe.same_calls.get(),
            1,
            "identical titles never probe; only `x` asked"
        );
    }

    #[test]
    fn magi_core_still_refuses_a_one_mage_consensus() {
        // [CP2 MS2 S2 loop 4] RV-S27 is declared unreachable while this holds: partial_trio needs a one-mage report.
        // If magi-core changes its default, this goes red and RV-S27 applies again.
        assert_eq!(
            magi_core::consensus::ConsensusEngine::default().min_agents(),
            2
        );
    }

    #[test]
    fn consolidated_findings_hold_only_redacted_model_text() {
        // [CP2 MS2 S2 loop 3] redaction at ingestion: no serializable struct keeps a credential a mage echoed.
        let f = serde_json::json!([{"severity":"warning","title":"t https://u:hunter2@example.invalid",
                                    "detail":"see https://user:hunter2@example.invalid/x","file":"a.rs","line":1}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        let all = serde_json::to_string(&c.findings).expect("serializes");
        assert!(!all.contains("hunter2"), "{all}");
    }

    #[test]
    fn one_mage_or_no_batch_is_a_consolidation_error() {
        // [CP2 MS2 S2a1 loop 10] the M-9 trigger T2.9 relies on (D14): magi-core refuses a one-agent consensus.
        let one = consolidate(&[batch(
            0,
            vec![out("melchior", "approve", 0.9, serde_json::json!([]))],
        )]);
        assert!(matches!(one, Err(ReviewError::Consolidation(_))));
        assert!(matches!(
            consolidate(&[]),
            Err(ReviewError::Consolidation(_))
        ));
    }

    #[test]
    fn a_per_mage_union_past_magi_cores_max_findings_is_consolidated_whole() {
        // [CP2 MS2 S2a1 loop 10] magi-core caps a mage at 100 findings when it validates a response (validate.rs:212),
        // never in `determine` (consensus.rs:253): three batches of 50 give a 150-finding union per mage, all kept.
        let many = |b: usize| {
            serde_json::Value::Array((0..50).map(|i| serde_json::json!(
            {"severity":"info","title":format!("t{b}-{i}"),"detail":"d","file":"a.rs","line":b * 100 + i + 1})).collect())
        };
        let batches: Vec<_> = (0..3)
            .map(|b| {
                batch(
                    b,
                    vec![
                        out("melchior", "approve", 0.9, many(b)),
                        out("caspar", "approve", 0.9, many(b)),
                    ],
                )
            })
            .collect();
        let c = consolidate(&batches).expect("consolidates");
        assert_eq!(c.findings.len(), 150);
        assert!(c.findings.iter().all(|f| f.sources.len() == 2));
    }

    #[test]
    fn a_zero_line_or_an_empty_file_groups_by_title_like_magi_core() {
        // [CP2 MS2 S2a1 loop 10] magi-core id-keys only a non-empty file with line > 0 (consensus.rs:203-216); the split
        // here is the same, so these group by title with both sources instead of falling to the WARN branch.
        let f = serde_json::json!([{"severity":"warning","title":"zero","detail":"d","file":"a.rs","line":0},
                                   {"severity":"warning","title":"empty","detail":"d","file":"","line":3}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 2);
        assert!(c.findings.iter().all(|x| x.sources.len() == 2));
    }

    #[test]
    fn redacting_model_text_twice_changes_nothing() {
        // [CP2 MS2 S2a1 loop 10] ingestion and write time both redact: the second pass must be a no-op, so the report and
        // the gate state show the same text.
        for raw in [
            "see https://user:hunter2@example.invalid/x",
            "two: https://a:b@h.invalid and postgres://u:p@db.invalid/x",
            "plain text",
        ] {
            let once = crate::redact::redact_foreign_text(raw).as_str().to_string();
            assert_eq!(
                crate::redact::redact_foreign_text(&once).as_str(),
                once,
                "{raw}"
            );
        }
        let f = serde_json::json!([{"severity":"warning","title":"t","detail":"see https://user:hunter2@example.invalid/x",
                                    "file":"a.rs","line":1}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, f.clone()),
                out("caspar", "approve", 0.9, f),
            ],
        )])
        .expect("consolidates");
        let d = &c.findings[0].details[0];
        assert_eq!(crate::redact::redact_foreign_text(d).as_str(), d.as_str());
    }

    #[test]
    fn agreement_counts_who_answered_in_the_batches_where_the_finding_appeared() {
        // [CP2 MS2 S2 loop 1] N-7 with real provenance: the finding appears only in batch 1, where Melchior alone answered.
        let crit = serde_json::json!([{"severity":"critical","title":"c","detail":"d","file":"b.rs","line":2}]);
        let mut c = consolidate(&[
            batch(
                0,
                vec![
                    out("melchior", "approve", 0.9, serde_json::json!([])),
                    out("balthasar", "approve", 0.9, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
            batch(1, vec![out("melchior", "reject", 0.8, crit)]),
        ])
        .expect("three mages across the review");
        assert_eq!(c.findings[0].batches, vec![1]);
        assert_eq!(c.answered[&1].len(), 1);
        let answered = c.answered.clone();
        apply_agreement(&mut c.findings, &answered);
        assert_eq!(
            (c.findings[0].severity, c.findings[0].marks.clone()),
            (Severity::Critical, vec![FindingMark::PartialTrio])
        );
    }

    #[test]
    fn an_empty_file_with_a_line_is_keyed_by_title_not_by_id() {
        // Mutation guard: magi-core id-keys only a non-empty file (consensus.rs:203-216). Two
        // different titles on `"" : 3` are two findings there; id-keying the empty file here
        // would merge them into one with two sources.
        let m = serde_json::json!([{"severity":"warning","title":"alpha","detail":"d","file":"","line":3}]);
        let c2 = serde_json::json!([{"severity":"warning","title":"beta","detail":"d","file":"","line":3}]);
        let c = consolidate(&[batch(
            0,
            vec![
                out("melchior", "approve", 0.9, m),
                out("caspar", "approve", 0.9, c2),
            ],
        )])
        .expect("consolidates");
        assert_eq!(c.findings.len(), 2, "{:?}", c.findings);
        assert!(
            c.findings.iter().all(|f| f.sources.len() == 1),
            "{:?}",
            c.findings
        );
    }

    #[test]
    fn a_worst_verdict_in_an_early_batch_outlives_a_later_approval() {
        // Mutation guard: the worst verdict wins wherever it appears, not the last batch's.
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let c = consolidate(&[
            batch(
                0,
                vec![
                    out("melchior", "reject", 0.6, serde_json::json!([])),
                    quiet(),
                ],
            ),
            batch(
                1,
                vec![
                    out("melchior", "approve", 0.9, serde_json::json!([])),
                    quiet(),
                ],
            ),
        ])
        .expect("consolidates");
        let m = c
            .agents
            .iter()
            .find(|a| a.agent == AgentName::Melchior)
            .expect("melchior");
        assert_eq!(m.verdict, Verdict::Reject);
        assert!((m.confidence - 0.6).abs() < 1e-9);
    }

    #[test]
    fn agreement_counts_every_batch_where_the_finding_appeared() {
        // Mutation guard: who answered is the union over ALL the finding's batches. Melchior
        // alone answered batch 0, all three answered batch 1, and only Melchior reported the
        // finding: two or more answered, so it is a single-mage finding.
        let f = serde_json::json!([{"severity":"critical","title":"c","detail":"d","file":"b.rs","line":2}]);
        let mut c = consolidate(&[
            batch(0, vec![out("melchior", "approve", 0.9, f.clone())]),
            batch(
                1,
                vec![
                    out("melchior", "approve", 0.9, f),
                    out("balthasar", "approve", 0.9, serde_json::json!([])),
                    out("caspar", "approve", 0.9, serde_json::json!([])),
                ],
            ),
        ])
        .expect("consolidates");
        assert_eq!(c.findings[0].batches, vec![0, 1]);
        let answered = c.answered.clone();
        apply_agreement(&mut c.findings, &answered);
        assert_eq!(
            (c.findings[0].severity, c.findings[0].marks.clone()),
            (Severity::Warning, vec![FindingMark::SingleMage])
        );
    }

    #[test]
    fn the_consensus_itself_holds_one_finding_when_categories_differ_on_a_line() {
        // Mutation guard: the neutral category reaches magi-core, so its consensus (the
        // report's verdict section) holds the one finding with both sources, like the list.
        let c = consolidate(&[batch(0, vec![
            out("melchior", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"a","detail":"da","file":"src/a.rs","line":10,"category":"logic-error"}])),
            out("caspar", "approve", 0.9, serde_json::json!([{"severity":"warning","title":"b","detail":"db","file":"src/a.rs","line":10,"category":"error-handling"}])),
        ])]).expect("consolidates");
        assert_eq!(c.consensus.findings.len(), 1, "{:?}", c.consensus.findings);
        assert_eq!(c.consensus.findings[0].sources.len(), 2);
    }

    #[test]
    fn per_mage_text_is_labelled_in_batch_order_redacted_and_follows_the_worst_batch() {
        // RV12: summary and reasoning of every batch, labelled and in batch order; the
        // recommendation from the worst-verdict batch; no model text kept unredacted.
        let secret = "https://user:hunter2@example.invalid/x";
        let mage = |verdict: &str, k: usize| {
            let mut o = out("melchior", verdict, 0.9, serde_json::json!([]));
            o.summary = format!("sum{k} {secret}");
            o.reasoning = format!("why{k} {secret}");
            o.recommendation = format!("rec{k} {secret}");
            o
        };
        let quiet = || out("caspar", "approve", 0.9, serde_json::json!([]));
        let c = consolidate(&[
            batch(0, vec![mage("approve", 1), quiet()]),
            batch(1, vec![mage("reject", 2), quiet()]),
        ])
        .expect("consolidates");
        let red = |s: String| crate::redact::redact_foreign_text(&s).as_str().to_string();
        let m = c
            .agents
            .iter()
            .find(|a| a.agent == AgentName::Melchior)
            .expect("melchior");
        assert_eq!(
            m.summary,
            format!(
                "[batch 1] {}\n[batch 2] {}",
                red(format!("sum1 {secret}")),
                red(format!("sum2 {secret}"))
            )
        );
        assert_eq!(
            m.reasoning,
            format!(
                "[batch 1] {}\n[batch 2] {}",
                red(format!("why1 {secret}")),
                red(format!("why2 {secret}"))
            )
        );
        assert_eq!(m.recommendation, red(format!("rec2 {secret}")));
        let all = serde_json::to_string(&c.agents).expect("serializes");
        assert!(!all.contains("hunter2"), "{all}");
    }

    #[test]
    fn a_file_without_a_line_is_grouped_by_probing_the_title_alone() {
        // The titles differ in case only, so membership needs a probe (with no probe allowed
        // they stay apart); the probe asks magi-core about the titles alone, never the files.
        let a = serde_json::json!([{"severity":"warning","title":"Leaks A Handle","detail":"da","file":"a.rs"}]);
        let b = serde_json::json!([{"severity":"warning","title":"leaks a handle","detail":"db","file":"b.rs"}]);
        let input = [batch(
            0,
            vec![
                out("melchior", "approve", 0.9, a),
                out("caspar", "approve", 0.9, b),
            ],
        )];
        let c = consolidate(&input).expect("consolidates");
        assert_eq!(
            c.consensus.findings.len(),
            1,
            "magi-core keys them by title"
        );
        assert_eq!(c.findings.len(), 1, "{:?}", c.findings);
        assert_eq!(
            c.findings[0].sources,
            vec![AgentName::Melchior, AgentName::Caspar]
        );
        assert_eq!(c.ungrouped_unlocated, 0);
        let unprobed = consolidate_with_probe_cap(&input, 0).expect("consolidates");
        assert_eq!(
            unprobed.findings.len(),
            2,
            "without a probe the second title is not claimed"
        );
    }

    #[test]
    fn a_member_magi_core_did_not_list_still_counts_as_a_source() {
        // [MAGI S6 loop 9, C1-m1] past magi-core's findings cap, Melchior's finding at a.rs:3
        // joins the group magi-core formed from Balthasar's alone: magi-core's `sources` says
        // [balthasar], and the group holds both mages' findings. Reading only magi-core's list
        // made the finding `single_mage` though two mages reported it.
        let f = |title: &str| -> Finding {
            serde_json::from_value(serde_json::json!({
                "severity": "warning", "title": title, "detail": "d",
                "file": "a.rs", "line": 3
            }))
            .expect("a Finding")
        };
        let reported = vec![
            Reported {
                batch: 0,
                agent: AgentName::Balthasar,
                finding: f("from balthasar"),
            },
            Reported {
                batch: 0,
                agent: AgentName::Melchior,
                finding: f("from melchior"),
            },
        ];
        let dedup: Vec<DedupFinding> = vec![serde_json::from_value(serde_json::json!({
            "severity": "warning", "title": "from balthasar", "detail": "d",
            "sources": ["balthasar"], "file": "a.rs", "line": 3, "category": "other"
        }))
        .expect("a DedupFinding")];
        let group = Group {
            dedup: Some(0),
            id: None,
            rep: "from balthasar".into(),
            members: vec![0, 1],
        };
        let c = finding_of(&group, &reported, &dedup, false);
        assert_eq!(c.sources, vec![AgentName::Balthasar, AgentName::Melchior]);
        assert_eq!(c.titles, vec!["from balthasar", "from melchior"]);
    }
}
