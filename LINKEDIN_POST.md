# LinkedIn post — ready to paste

Attach **`demo/self_play.gif`** (or a screen recording of you playing it) to whichever
version you use. The GIF is what stops the scroll — lead with it.

---

## Version A — the build story (recommended)

> I taught a neural network to play chess by showing it grandmaster games. ♟️
>
> No hand-written rules. No opening book. No search engine like Stockfish. Just a
> convolutional neural network that looks at a board and predicts the move a
> grandmaster would play — a technique called *behavioral cloning* (the same idea
> behind the Maia research on human-like chess AI).
>
> How it works, end to end:
> • Every board position is encoded as a 17×8×8 tensor (piece locations, whose turn,
>   castling rights).
> • A residual CNN (128 channels × 10 layers) maps that to one of 4,096 possible moves.
> • Trained on thousands of real grandmaster games on a GPU; it runs on a plain CPU.
> • At inference I mask the output to legal moves only — so it *never* plays an
>   illegal move.
>
> Is it going to beat Stockfish? No — and that's the honest part. Because it has no
> search, it plays solid, human-like chess but can still miss a tactic. What it shows
> is the full ML-engineering loop: data pipeline → encoding → model → training →
> deployment.
>
> Code + notebook are open source, and you can **play it live in your browser** 👇
> 🎮 Play: https://huggingface.co/spaces/Ahmed007/chess-expert
> 💻 Code: [GitHub link]
>
> Next up: deeper search so it doesn't just *imitate* grandmasters but starts to
> *outplay* opponents.
>
> #MachineLearning #DeepLearning #Chess #PyTorch #AI #ComputerScience

---

## Version B — short and punchy

> I trained a neural network to play chess from grandmaster games — no rules, no
> search, just deep learning. ♟️
>
> It encodes the board as a tensor, a residual CNN predicts the move a grandmaster
> would make, and legal-move masking guarantees it never plays an illegal move.
> Trains on a GPU, plays on a CPU.
>
> Honest take: it imitates grandmasters, so it plays human-like chess with only
> shallow search — so the occasional blunder. A clean end-to-end ML project.
> 🎮 Play it live: https://huggingface.co/spaces/Ahmed007/chess-expert  · 💻 Code: [link]
>
> #DeepLearning #MachineLearning #Chess #PyTorch #AI

---

## Tips for reach

- **Let people play it** — the Hugging Face Space link lets readers try it in one
  click, which drives comments and shares. Pin it in the first comment too.
- **Post the GIF, not a screenshot** — motion gets far more engagement.
- Put the GitHub link in the **first comment** as well as the post (LinkedIn softly
  down-ranks posts with outbound links in the body).
- Ask a question at the end to invite comments, e.g. *"Would you add search next, or
  more data?"*
- Since this supports master's applications: keep the honest framing. Reviewers
  respect "here's exactly what it does and doesn't do" far more than "GM-level AI."
- First line matters most — it's what shows before "…see more". Lead with the hook.
