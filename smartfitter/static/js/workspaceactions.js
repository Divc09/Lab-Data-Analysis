// One routing table serves the toolbar, command palette and keyboard shortcuts.
// Callers supply capabilities and handlers; no workspace falls back to Analyze.
const ROUTES = {
  analyze: [['analysis', 'Save analysis', 'Export the active Analyze file using its Export-tab options'], ['analysis', 'Copy figure', 'Copy the Analyze figure']],
  figure: [['figure', 'Export figure', 'Choose SVG, PNG or PDF for the assembled figure'], ['figure', 'Copy figure', 'Copy the assembled Figure as PNG']],
  compare: [['trend', 'Export trend', 'Save the current Compare trend table as CSV'], ['overlay', 'Copy overlay', 'Copy the current Compare overlay figure']],
  day: [['day', 'Export day CSV', 'Save the displayed Day summary rows as CSV'], ['day', 'Copy day table', 'Copy the displayed Day summary rows']],
  stack: [['stack', 'Export stack CSV', 'Save the Stacks results and provenance as CSV'], ['stack', 'Copy stack table', 'Copy the Stacks results table']],
  batch: [['batch', 'Export batch CSV', 'Save all loaded Batch summary rows as CSV'], ['batch', 'Copy batch table', 'Copy all loaded Batch summary rows']],
  autofind: [['autofind', 'Export figures', 'Export the selected AutoFind runs as a figure set'], null],
  home: [['session', 'Save session', 'Save the open workspace as a session'], null],
};

export function workspaceActions(view, capabilities, handlers) {
  const route = ROUTES[view] || ROUTES.home;
  return Object.fromEntries(['save', 'copy'].map((action, i) => {
    const entry = route[i];
    if (!entry) return [action, null];
    const [key, label, title] = entry;
    return [action, { label, title, enabled: !!capabilities[key], run: handlers[`${key}_${action}`] }];
  }));
}
