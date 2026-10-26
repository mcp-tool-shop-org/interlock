import type { SiteConfig } from '@mcptoolshop/site-theme';

export const config: SiteConfig = {
  title: 'interlock',
  description:
    'Local AI device router: it sends each job to a model, an engine and a device named by a measured receipt.',
  logoBadge: 'IL',
  brandName: 'interlock',
  repoUrl: 'https://github.com/mcp-tool-shop-org/interlock',
  footerText:
    'MIT Licensed — built by <a href="https://mcp-tool-shop.github.io/" style="color:var(--color-muted);text-decoration:underline">mcp-tool-shop</a>',

  hero: {
    badge: 'v1.0.6',
    headline: 'interlock',
    headlineAccent: 'local AI device router.',
    description:
      'A research instrument. It suggests a model, an engine and a device from measured receipts, and it never serves traffic.',
    primaryCta: {
      href: 'https://github.com/mcp-tool-shop-org/interlock',
      label: 'View on GitHub',
    },
    secondaryCta: { href: 'handbook/', label: 'Read the Handbook' },
    previews: [
      {
        label: 'Install from a clone',
        code: 'uv sync\nuv run interlock --help',
      },
      {
        label: 'Ask the table',
        code: 'uv run interlock route --task generate --model Phi --thinking off --ctx 1024 --busy 5090',
      },
      {
        label: 'Plan a run',
        code: 'uv run interlock run experiments/e4b/spec.yaml --dry-run',
      },
    ],
  },

  sections: [
    {
      kind: 'features',
      id: 'features',
      title: 'What it does',
      subtitle: 'Every number comes from a receipt. A route is a suggestion.',
      features: [
        {
          title: 'Named devices',
          desc: 'NPU, the Intel iGPU, CPU, and the RTX 5090. A missing device is an error. interlock does not fall off onto a weaker one.',
        },
        {
          title: 'The conservative bound',
          desc: 'Generate ranks by the worst repeat, not the median. When no row meets the constraints, the command exits 1 and names the constraint.',
        },
        {
          title: 'Pre-registered experiments',
          desc: 'A spec is committed before any device time. interlock run refuses a spec that was edited after its commit.',
        },
        {
          title: 'One rig',
          desc: 'The table was measured on one machine: a Core Ultra 9 285K with its NPU and iGPU, plus an RTX 5090. Treat a route as that rig’s evidence.',
        },
      ],
    },
    {
      kind: 'data-table',
      id: 'devices',
      title: 'Devices',
      subtitle: 'Every device is named. OpenVINO AUTO, HETERO and MULTI are refused.',
      columns: ['Name', 'What it is'],
      rows: [
        ['NPU', 'Intel AI Boost on the Core Ultra 9 285K. Exclusive. A step needs --npu-session.'],
        ['iGPU', 'The Intel integrated GPU. A live enumeration must say vendor 0x8086 and type INTEGRATED.'],
        ['CPU', 'The same processor’s CPU. A CPU-heavy leg waits for --cpu-window.'],
        ['5090', 'The RTX 5090, through local Ollama. A step needs --grant.'],
      ],
    },
    {
      kind: 'code-cards',
      id: 'usage',
      title: 'From a clone',
      subtitle: 'These commands do not start a device.',
      cards: [
        {
          title: 'Check the receipts',
          code: 'uv run interlock check',
        },
        {
          title: 'Read the table',
          code: 'uv run interlock table --task generate --device NPU',
        },
        {
          title: 'List runs',
          code: 'uv run interlock runs list',
        },
      ],
    },
    {
      kind: 'api',
      id: 'commands',
      title: 'Commands',
      subtitle: 'The handbook has the flags. Nothing here is a server.',
      apis: [
        {
          signature: 'interlock route',
          description: 'Choose a model, engine and device from the routing table. Advisory only.',
        },
        {
          signature: 'interlock run',
          description: 'Run a committed spec. --dry-run prints the plan and starts nothing.',
        },
        {
          signature: 'interlock runs',
          description: 'List complete, aborted and stray runs, or set a stray series aside.',
        },
        {
          signature: 'interlock table',
          description: 'Print routing rows, filtered by task or device.',
        },
        {
          signature: 'interlock check',
          description: 'Validate every receipt and table row, including the privacy check.',
        },
        {
          signature: 'interlock import',
          description: 'Rebuild table/ from the committed receipts.',
        },
      ],
    },
  ],
};
