import { LayoutDashboard, Search, GitBranch, BookOpen, Bell, Shield, LogOut, Activity } from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard',  icon: LayoutDashboard, label: 'Dashboard' },
  { id: 'investigate', icon: Search,         label: 'Investigate' },
  { id: 'graph',       icon: GitBranch,      label: 'Graph View' },
  { id: 'rag',         icon: BookOpen,       label: 'Regulations' },
  { id: 'alerts',      icon: Bell,           label: 'Alerts' },
  { id: 'mlops',       icon: Activity,       label: 'MLOps' },
];

export default function Sidebar({ page, setPage }) {
  return (
    <aside className="sidebar">
      <div className="sidebar-logo">
        <Shield size={20} color="white" />
      </div>

      <nav className="sidebar-nav">
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => (
          <div key={id} className="tooltip-wrap" style={{ width: '100%' }}>
            <button
              className={`sidebar-btn ${page === id ? 'active' : ''}`}
              onClick={() => setPage(id)}
              title={label}
            >
              <Icon />
            </button>
            <span className="tooltip-box" style={{ left: 'calc(100% + 12px)', bottom: 'auto', top: '50%', transform: 'translateY(-50%)' }}>
              {label}
            </span>
          </div>
        ))}
      </nav>

      <div className="sidebar-footer">
        <button className="sidebar-btn" title="Logout" style={{ color: 'var(--text-muted)' }}>
          <LogOut />
        </button>
      </div>
    </aside>
  );
}
