import solara
from solara.website.utils import apidoc
import importlib.util
from pathlib import Path

# Helper to dynamically load Solara apps
def load_solara_page(module_path):
    spec = importlib.util.spec_from_file_location("app", module_path)
    app_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app_module)
    if not hasattr(app_module, "Page"):
        raise AttributeError(f"Module at {module_path} does not have a 'Page' function.")
    return app_module.Page

# Define paths to each application's app.py
recsys_home_app_path = Path("application/website/01-ml-recsys-content-based.py").resolve()
recsys_createModel_app_path = Path("application/website/02-ml-create-model.py").resolve()
recsys_useModel_app_path = Path("application/website/03-ml-display-rec.py").resolve()

# Load Solara apps
HomeApp = load_solara_page(recsys_home_app_path)
CreateModelApp = load_solara_page(recsys_createModel_app_path)
UseModelApp = load_solara_page(recsys_useModel_app_path)

GTM_ID = "GTM-W4MNT473"
GTM_HEAD_SNIPPET = f"""(function(w,d,s,l,i){{w[l]=w[l]||[];w[l].push({{'gtm.start':
new Date().getTime(),event:'gtm.js'}});var f=d.getElementsByTagName(s)[0],
j=d.createElement(s),dl=l!='dataLayer'?'&l='+l:'';j.async=true;j.src=
'https://www.googletagmanager.com/gtm.js?id='+i+dl;f.parentNode.insertBefore(j,f);
}})(window,document,'script','dataLayer','{GTM_ID}');"""
GTM_NOSCRIPT_SNIPPET = f'<iframe src="https://www.googletagmanager.com/ns.html?id={GTM_ID}" height="0" width="0" style="display:none;visibility:hidden"></iframe>'

# Routes for the central menu
routes = [
    solara.Route(path="/", component=HomeApp, label="LIKYLY - Solution IA de recommandations de produits/oeuvres/articles - demo"),
    solara.Route(path="use-the-IA-Model", component=UseModelApp,
                 label="LIKYLY - Solution IA de recommandations de produits/oeuvres/articles - demo"),
    solara.Route(path="evaluate-the-IA-model", component=CreateModelApp, label="LIKYLY - Solution IA de recommandations de produits/oeuvres/articles - demo"),

]


LABELS = {
    "/": "Accueil",
    "model": "LIKYLY – Démo recommandations",
    "Ceci est un test": "LIKYLY – Solution IA de recommandations de produits/œuvres/articles - demo",
    # ajoute ici les autres routes si besoin
}

@solara.component
def Page():
    route_current, routes = solara.use_route()
    with solara.VBox() as main:
        with solara.Head():
            solara.HTML(tag="script", unsafe_innerHTML=GTM_HEAD_SNIPPET)
        solara.HTML(tag="noscript", unsafe_innerHTML=GTM_NOSCRIPT_SNIPPET)
        solara.Info("Note the address bar in the browser. It should change to the path of the link.")
        with solara.HBox():
            for route in routes:
                with solara.Link(route):
                    current = route_current is route
                    # On privilégie le label de la route, sinon fallback sur le path
                    label = getattr(route, "label", None) or f"Go to {route.path}"
                    solara.Button(label, color="red" if current else None)
    return main


