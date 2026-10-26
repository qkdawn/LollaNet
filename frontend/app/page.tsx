import { Suspense } from 'react';
import Workbench from '../components/workbench';
export default function Page() { return <Suspense fallback={<main className="loading">LollaNet</main>}><Workbench /></Suspense>; }
