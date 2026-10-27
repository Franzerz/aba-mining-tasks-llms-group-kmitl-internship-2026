# Benchmark port (parked, not wired up)

This folder holds the MySQL-backed reimplementation of mu_team's `Project_ABA` — the version
that was verified byte-identical to their app (16/16 in-process graph diffs, 27/27 live HTTP
diffs) and had the Cytoscape renderer.

It was moved aside when the project was reverted to the standalone Task-3-gold version. Nothing
here is imported by the running app; it is kept because there is no git history in this repo to
recover it from.

To bring it back, move these files to their original locations:

    app/{db,normalizers,settings}.py
    app/services/*
    app/routes/api.py
    benchmark/*
    frontend/src/graph/displayModel.js
    frontend/src/components/{CytoscapeGraph,NewAbaPanel}.jsx
    frontend/src/vendor/*

and restore `app/main.py` to include `routes.api`, plus the MySQL/Redis deps in
`requirements.txt` (PyMySQL, redis, httpx, python-dotenv). It also needs the `aba-mysql` and
`aba-redis` containers running.
