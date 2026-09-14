import { cn } from "../lib/cn";

export function Logo({ className }: { className?: string }) {
  return (
    <div className={cn("flex items-center gap-2.5", className)}>
      <span className="relative inline-flex h-7 w-7 items-center justify-center rounded-[9px] bg-gradient-to-br from-brand-400 via-brand-500 to-accent-600 shadow-[0_8px_24px_-6px_rgba(70,86,245,0.6)]">
        <span className="absolute inset-[1px] rounded-[8px] bg-ink-950/40" />
        <span className="relative font-display text-[13px] font-semibold tracking-tight text-white">
          r
        </span>
      </span>
      <span className="font-display text-[15px] font-semibold tracking-tight text-white">
        regulo
      </span>
    </div>
  );
}
