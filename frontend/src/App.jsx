import { useState } from 'react';
import Sidebar     from './components/Sidebar';
import Dashboard   from './pages/Dashboard';
import Investigate from './pages/Investigate';
import GraphView   from './pages/GraphView';
import RAGLookup   from './pages/RAGLookup';
import Alerts      from './pages/Alerts';
import MLOps       from './pages/MLOps';
import './index.css';

export default function App() {
  const [page, setPage]                   = useState('dashboard');
  const [initialAccount, setInitialAccount] = useState('');

  function renderPage() {
    switch (page) {
      case 'dashboard':
        return <Dashboard setPage={setPage} setInitialAccount={setInitialAccount} />;
      case 'investigate':
        return <Investigate initialAccount={initialAccount} setInitialAccount={setInitialAccount} />;
      case 'graph':
        return <GraphView />;
      case 'rag':
        return <RAGLookup />;
      case 'alerts':
        return <Alerts setPage={setPage} setInitialAccount={setInitialAccount} />;
      case 'mlops':
        return <MLOps />;
      default:
        return <Dashboard setPage={setPage} setInitialAccount={setInitialAccount} />;
    }
  }

  return (
    <div className="app-shell">
      <Sidebar page={page} setPage={setPage} />
      {renderPage()}
    </div>
  );
}
