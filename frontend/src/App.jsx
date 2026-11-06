import { PrefsProvider } from "./state/prefs.jsx";
import { RouterProvider, useRouter } from "./state/router.jsx";
import { SessionProvider } from "./state/session.jsx";
import Sparkles from "./components/Sparkles.jsx";
import TopBar from "./components/TopBar.jsx";
import Home from "./pages/Home.jsx";
import Workspace from "./pages/Workspace.jsx";

function Pages() {
  const { path } = useRouter();
  return (
    <>
      <div className="bg-glow" aria-hidden="true" />
      <Sparkles />
      <TopBar />
      {path === "/app" ? <Workspace /> : <Home />}
    </>
  );
}

export default function App({ client }) {
  return (
    <PrefsProvider>
      <RouterProvider>
        <SessionProvider client={client}>
          <Pages />
        </SessionProvider>
      </RouterProvider>
    </PrefsProvider>
  );
}
