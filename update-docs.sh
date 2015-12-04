#!/bin/bash
stack haddock
cd doc
git rm -rf .
cp -r $(stack path --local-doc-root)/* ./
git add --all
git commit -a -m "Automatic documentation update."
git push origin gh-pages
cd ..

