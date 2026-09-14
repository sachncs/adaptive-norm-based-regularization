import { motion } from "framer-motion";

type Penalty = {
  name: string;
  formula: string;
  params: string;
  category: "Classical" | "Adaptive";
  notes: string;
  gradient: string;
};

const penalties: Penalty[] = [
  {
    name: "Void",
    formula: "0",
    params: "—",
    category: "Classical",
    notes: "No penalty. The reference point every other regularizer is measured against.",
    gradient: "∇0 = 0",
  },
  {
    name: "Ridge",
    formula: "λ ‖W‖²_F",
    params: "λ",
    category: "Classical",
    notes: "L2 weight decay. Shrinks every weight uniformly toward zero.",
    gradient: "2λW",
  },
  {
    name: "Lasso",
    formula: "γ ‖W‖₁",
    params: "γ",
    category: "Classical",
    notes: "L1 sparsity. Drives irrelevant weights to exactly zero.",
    gradient: "γ · sign(W)",
  },
  {
    name: "ElasticNet",
    formula: "αγ ‖W‖₁ + (1−α)/2 ‖W‖²_F",
    params: "α, γ",
    category: "Classical",
    notes: "Mix of L1 and L2. Sparsity with shrinkage stability.",
    gradient: "αγ·sign(W) + (1−α)λW",
  },
  {
    name: "Covridge",
    formula: "λ₁ ‖C^½ W‖²_F + λ₂ ‖W‖²_F",
    params: "λ₁, λ₂, gram",
    category: "Adaptive",
    notes: "Geometry-aware shrinkage along the eigenvectors of the empirical Gram matrix.",
    gradient: "C W · 2λ₁ + 2λ₂W",
  },
  {
    name: "Sparridge",
    formula: "λ₁ ‖C^½ W‖²_F + γ ‖W‖₁",
    params: "λ₁, γ, gram",
    category: "Adaptive",
    notes: "Covridge's sparse cousin. Adaptive shrinkage with selective sparsity.",
    gradient: "C W · 2λ₁ + γ·sign(W)",
  },
];

export function Penalties() {
  return (
    <section id="penalties" className="relative py-28 sm:py-36">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-gradient-to-b from-transparent via-brand-500/[0.04] to-transparent" />
      <div className="container-x">
        <div className="mx-auto max-w-2xl text-center">
          <div className="inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 font-mono text-[11px] uppercase tracking-[0.18em] text-ink-300">
            <span className="h-1 w-1 rounded-full bg-accent-400" />
            Penalties
          </div>
          <h2 className="mt-6 font-display text-[32px] font-semibold leading-[1.1] tracking-tight text-white sm:text-[44px] text-balance">
            Six penalties.{" "}
            <span className="text-ink-400">One interface.</span>
          </h2>
          <p className="mt-5 text-[17px] leading-relaxed text-ink-300 text-pretty">
            Every penalty exposes{" "}
            <span className="font-mono text-ink-100">value(W)</span> and{" "}
            <span className="font-mono text-ink-100">grad(W)</span>. Drop one in.
            Swap one out. No learning curve.
          </p>
        </div>

        <div className="mt-16 grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-3">
          {penalties.map((p, i) => (
            <motion.div
              key={p.name}
              initial={{ opacity: 0, y: 16 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{
                duration: 0.55,
                delay: i * 0.05,
                ease: [0.22, 1, 0.36, 1],
              }}
              className="group relative flex flex-col overflow-hidden rounded-2xl border border-white/[0.06] bg-ink-900/40 p-6 transition-colors hover:border-white/10 hover:bg-ink-900/60"
            >
              <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-white/10 to-transparent" />

              <div className="flex items-center justify-between">
                <span
                  className={`font-mono text-[10px] uppercase tracking-[0.18em] ${
                    p.category === "Adaptive"
                      ? "text-accent-300"
                      : "text-ink-400"
                  }`}
                >
                  {p.category}
                </span>
                {p.category === "Adaptive" && (
                  <span className="inline-flex items-center gap-1 rounded-full border border-accent-400/20 bg-accent-400/10 px-1.5 py-0.5 font-mono text-[9px] uppercase tracking-wider text-accent-200">
                    <span className="h-1 w-1 rounded-full bg-accent-400" />
                    new
                  </span>
                )}
              </div>

              <h3 className="mt-3 font-display text-[20px] font-semibold tracking-tight text-white">
                {p.name}
              </h3>

              <div className="mt-4 rounded-lg border border-white/[0.05] bg-ink-950/60 px-3.5 py-3 font-mono text-[14px] leading-relaxed text-ink-100">
                {p.formula}
              </div>

              <div className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 border-t border-white/[0.05] pt-4">
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
                    params
                  </div>
                  <div className="mt-0.5 font-mono text-[12px] text-ink-200">
                    {p.params}
                  </div>
                </div>
                <div>
                  <div className="font-mono text-[10px] uppercase tracking-wider text-ink-500">
                    gradient
                  </div>
                  <div className="mt-0.5 truncate font-mono text-[12px] text-ink-200">
                    {p.gradient}
                  </div>
                </div>
              </div>

              <p className="mt-4 text-[13.5px] leading-relaxed text-ink-300">
                {p.notes}
              </p>
            </motion.div>
          ))}
        </div>

        <p className="mt-10 text-center font-mono text-[12px] text-ink-500">
          Covridge & Sparridge operate on the first weight matrix where the
          empirical Gram matrix is defined.
        </p>
      </div>
    </section>
  );
}
