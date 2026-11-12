# Review history

A PR often has reviews, threads and new commits before you arrive. Read them before looking for
problems, so the review builds on the discussion instead of repeating it.

## Fetch everything in one query

```bash
gh api graphql -F n=<N> -f query='
query($n:Int!){repository(owner:"NCAR",name:"miles-credit"){pullRequest(number:$n){
  headRefOid reviewDecision lastEditedAt
  reviews(first:100){nodes{author{login} state submittedAt commit{oid} body}}
  reviewThreads(first:100){nodes{isResolved isOutdated path line
    comments(first:50){nodes{author{login} createdAt body}}}}
  comments(first:100){nodes{author{login} createdAt body}}
  commits(last:100){nodes{commit{oid committedDate messageHeadline}}}
  userContentEdits(first:20){nodes{editedAt editor{login}}}
  timelineItems(first:100,itemTypes:[HEAD_REF_FORCE_PUSHED_EVENT,REVIEW_REQUESTED_EVENT,RENAMED_TITLE_EVENT]){nodes{
    __typename
    ... on HeadRefForcePushedEvent{createdAt}
    ... on ReviewRequestedEvent{createdAt requestedReviewer{... on User{login}}}
    ... on RenamedTitleEvent{previousTitle currentTitle}}}
}}}'
gh pr checks <N>
```

Skip bot boilerplate (e.g. the `github-actions` review checklist), but note which of its boxes are still open.

## Build these from it

1. **Rounds.** Each review: who, verdict, date, and the commit it reviewed (`commit.oid`). Who is requested
   but has not reviewed yet.
2. **Ledger of points raised.** One line per point, from review bodies, inline threads and comments:
   who raised it → what the author did (commit that addressed it, reply, nothing) → status:
   - **Addressed**: verify it in the current code (a mutation test or a quick check), don't trust "done".
   - **Open**: still unanswered or unresolved.
   - **Declined**: the author pushed back. Summarize both positions neutrally.
   - **Superseded**: the code it referred to is gone (`isOutdated`).
3. **What changed since the last review.** Commits after the last review's commit, and their diff
   restricted to the PR's files: `git diff <reviewed-oid> HEAD -- <files>`. Merges from the base branch
   also show up here; set aside what came from the base (`git log --first-parent`). If the branch was
   force-pushed after a review, the reviewed commit may be gone: say so and review the whole PR.
4. **Scope drift.** Compare the original description and linked issue with what the PR does now. Edited
   descriptions (`userContentEdits`) and fix commits often move scope; note any that were not discussed.
5. **Where people agree and disagree.** Reviewer vs author, and reviewer vs reviewer. Points where they
   disagree, or questions nobody answered (e.g. "was this run on Casper?"), are the candidates for
   **Needs your call**.

## Use it in the review

- **Don't repeat** a point already raised. If you have new evidence about it, reply in that thread, or
  mention it in the summary with credit ("djgagne's item 1, now fixed — verified by …").
- **Focus on the delta.** When the user or anyone else already reviewed, look hardest at the commits after
  that review: fixes made in response to a review are new, often unreviewed code.
- **Re-review.** If the current `gh` user reviewed before, check each of their own points (addressed? how?)
  before anything else, and keep the new review to what changed.
- **Explain stale text.** A PR description that no longer matches the code after review-driven changes is a
  routine "please update the description", not a mismatch. Say which change made it stale.
