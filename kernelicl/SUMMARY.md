# What we built, and why

A plain-language summary of this work. No machine-learning background assumed.
For how to run any of it, see [README.md](README.md).

---

## The problem

Suppose you have a model that looks at a patient's screening data and predicts
whether they should be referred. Modern models are very good at this. They are also
**black boxes**: they produce an answer and no account of how they reached it.

That is a real obstacle in healthcare. A clinician cannot check reasoning they cannot
see. A regulator cannot approve a decision nobody can explain. And when the model is
wrong, nobody finds out why.

The usual workaround is to explain the model *after the fact* — tools like SHAP that
attribute a prediction to the input features. The trouble is that these explanations
are reconstructions. They are a plausible story about what the model might have done,
and they can be wrong about it.

## The idea

A 2026 paper, *Interpretable Tabular Foundation Models via In-Context Kernel
Regression*, proposes something different: **change the model so that the explanation
is the calculation.**

The model normally ends with a neural network layer that turns its internal
representation into an answer. That layer is where the reasoning disappears. Replace
it with something simpler — a **weighted average of past cases**:

```
prediction  =  w₁ × (outcome of patient 1)
             + w₂ × (outcome of patient 2)
             + …
```

Every past patient in your training data gets a weight. Similar patients get large
weights, dissimilar ones get roughly zero. The prediction is the weighted vote.

The weights are not an estimate of influence. They are the actual numbers in the sum.
So "this patient was referred because of these five similar past cases" is not an
interpretation — it is a readout of the arithmetic. It cannot be unfaithful, because
there is nothing else happening.

**The cost is small.** The paper reports about 0.2 accuracy points across 55 benchmark
datasets. What you get back is a named set of past cases behind every prediction.

The paper published no code. Everything here is an implementation of its equations
from scratch.

## What was built

### Changes to the model itself

Three additions to the TabICL package, covered by 155 passing tests.

The subtle one is called **symmetric embeddings**. The model treats training patients
and new patients differently — training patients are "reference material," new
patients are "questions." That difference means the notion of distance between them is
not well defined, which breaks the whole idea of "similar patients." The fix is to run
the training patients through a second time, in the role of questions, so both sit in
the same frame of reference and distances mean something. It costs roughly twice the
computation in that stage.

### Six tools

Each is a standalone file you paste into a notebook.

| File | What it gives you |
|---|---|
| **quickstart** | the mechanics in 60 lines, no abstraction |
| **clinical** | the core: per-case evidence, triage, auditing, fairness checks |
| **analysis** | tables and charts comparing kernels, and against plain TabICL |
| **embeddings** | what the model's internal map looks like |
| **shift** | corruption testing — does the model notice when its input is wrong? |
| **finetune** | retrains the model so distances mean similarity |

### What you can now do

**1. Explain a single decision.** For any patient, list the past cases behind the
prediction, by record ID, with how much each contributed. A clinician can pull those
records and check whether the comparison is reasonable.

**2. Triage the workload.** Every prediction carries an *evidence base* — how many past
cases it effectively rests on — and an *agreement* score. Predictions resting on thin
or conflicting evidence get flagged for human review. In testing this split a cohort
into roughly 95% routine, 4% review, 1% requiring a clinician.

**3. Detect when the model is out of its depth.** If a new patient is unlike anything
in the training data, the model says so instead of guessing confidently.

**4. Find bad training data.** Records that repeatedly sit behind *wrong* answers are
likely mislabelled. On a test where we deliberately corrupted 60 labels, this recovered
41 of them in the top 60 candidates, against 5 expected by chance.

**5. Check fairness properly.** For any group — site, age band, sex — you can ask
whether predictions for that group draw on that group's data or are extrapolated from
elsewhere. In one test a group was 53% of the caseload but drew only **16%** of its
evidence from its own records, while showing identical accuracy. No standard fairness
metric would have caught that.

**6. See what the model thinks "similar" means.** Which measurements does it insist
comparable patients agree on? Take that list to a clinician. If it matches domain
knowledge, that is independent evidence the model learned something real. If it has
latched onto an administrative field like a site code, you have caught a model that
will collapse at a new clinic.

## Things we found along the way

Most of these contradicted a reasonable assumption. They are recorded so nobody has to
rediscover them.

**Training patients can see their own answers.** Because a training patient sits inside
the model's own reference material, its internal representation contains its own
outcome. A simple test recovered training outcomes from these representations perfectly
(100%). This is by design and does not affect predictions for new patients — but it
means any quality check run on training data is meaningless. Everything is now measured
on held-out data instead.

**Almost all the useful structure comes from one stage.** The model has three stages.
The first two, which process the features, leave patients no better separated by
outcome than the raw spreadsheet does (0.829 versus 0.829 on a like-for-like measure).
The third stage — the one that compares each patient against labelled examples — is
where separation appears (0.972). Useful to know: that is the stage worth investing in.

**Tuning for accuracy alone produces useless explanations.** The model has a dial
controlling how many past cases each prediction draws on. Accuracy barely changes
across most of its range, while the number of cases varies enormously. Tuning purely
for accuracy therefore picks a setting where each prediction averages over the *entire*
training set — technically accurate, and explaining nothing. The tuning now accepts up
to one accuracy point of loss in exchange for a far smaller, readable evidence base.

**Finding bad labels requires ignoring an instinct.** The obvious approach is to only
examine records with substantial influence. That is backwards: a record whose recorded
outcome contradicts its neighbours gets pushed *away* by the model, so it carries less
weight than average. Filtering on influence discarded exactly the records being hunted,
cutting recovery from 41-of-60 to 5-of-60.

**Accuracy is the wrong headline metric for screening.** On a cohort that was 85% one
outcome, some settings scored 0.85 accuracy by predicting the majority for
*everybody*. A balanced measure exposed them immediately as 0.50 — no better than a
coin flip. The tooling now uses the balanced measure by default.

**Corruption testing should measure detection, not damage.** If you deliberately
corrupt an input to test robustness, the standard result is an accuracy drop. That
tells you the model is sensitive but not whether it *knew*. A model that degrades while
flagging its new errors is behaving well; one that degrades while staying confident is
dangerous. In one test accuracy barely moved (0.986 → 0.971) while the detection rate
halved — errors becoming invisible, which the accuracy figure entirely concealed.

**Several bugs only appeared on real-shaped data.** Passing data as a spreadsheet
rather than a plain array silently reported the *wrong patient records* next to the
weights — the worst possible failure for a tool whose job is attribution. Others: the
code crashed on text columns, and on any machine with a GPU it crashed when asked to
run on the processor. All were invisible on tidy numeric test data.

## Where this stands

**Solid.** The mechanism is implemented and tested (155 tests). All six tools run
end-to-end on realistic messy data — spreadsheets with text columns, missing values,
more than two outcomes. Everything above was measured, not assumed.

**Not established.** Every number quoted here comes from **synthetic data**, generated
to exercise the code. None of it says anything about your dataset. The paper's own
results have not been reproduced, and since the paper published no code, agreement with
it is unverified.

**Not clinical software.** This is an analysis tool for deciding whether the approach is
worth pursuing. Any use affecting patients would need prospective validation, thresholds
set against real reviewer capacity, and governance that does not exist here.

**Two honest limits of the method itself.** A faithful explanation is not a correct one
— a wrong prediction gets a clean, readable, wrong explanation. And the explanation is
about similarity, not cause: "resembles these past cases," never "because of this
measurement."

## What happens next

1. **Run the tools on the real dataset.** The single most informative output is a chart
   comparing accuracy against explainability, which tells you how few past cases a
   prediction can rest on before accuracy suffers.

2. **Retraining is underway.** Out of the box, the model's internal map was built for
   its original purpose, not for measuring similarity. Retraining reshapes it. This is
   a one-off cost of a few hours on a suitable machine — it learns from synthetic data
   and never sees your patients, so the result is reusable on any dataset, forever.

3. **Then compare.** Two numbers in the embeddings tool should improve if the retraining
   worked. If they do, the explanations get sharper for free.

4. **Take the "what does similar mean" list to a clinician.** That is the cheapest step
   with the highest chance of telling you something true about the data rather than
   about the method.
