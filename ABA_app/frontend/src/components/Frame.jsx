// The rounded, bordered content slab mu_team wrapped every page in (.frame).
export default function Frame({ children, className = "" }) {
  return (
    <main
      className={`w-[min(1760px,calc(100vw-24px))] mx-auto my-6 p-7 border border-edge rounded-[22px]
                  bg-gradient-to-b from-section to-[#ece3d7] shadow-frame ${className}`}
    >
      {children}
    </main>
  );
}

export function PageHead({ eyebrow, title, subtitle, actions }) {
  return (
    <header className="flex justify-between items-start gap-4 flex-wrap">
      <div className="max-w-[76ch]">
        {eyebrow && <p className="eyebrow m-0">{eyebrow}</p>}
        <h1 className="mt-1.5 mb-0 text-[clamp(30px,3vw,48px)] leading-[1.03] tracking-[-0.02em]">
          {title}
        </h1>
        {subtitle && <p className="mt-2.5 mb-0 text-[13px] text-muted leading-[1.45]">{subtitle}</p>}
      </div>
      {actions && <div className="flex items-center gap-2.5">{actions}</div>}
    </header>
  );
}
