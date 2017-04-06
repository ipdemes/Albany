#!/bin/bash

source ./env-all.sh

cd "$LCM_DIR"

# Clone package repositories.
for PACKAGE in $PACKAGES; do
    case "$PACKAGE" in
	trilinos)
	    PACKAGE_NAME="Trilinos"
	    REPO="git@github.com:trilinos/Trilinos.git"
	    ;;
	albany)
	    PACKAGE_NAME="Albany"
	    REPO="git@github.com:gahansen/Albany.git"
	    ;;
	*)
	    echo "Unrecognized package option"
	    exit 1
	    ;;
    esac
    PACKAGE_DIR="$LCM_DIR/$PACKAGE_NAME"
    CHECKOUT_LOG="$PACKAGE-checkout.log"
    if [ -d "$PACKAGE_DIR" ]; then
	rm "$PACKAGE_DIR" -rf
    fi
    git clone -v "$REPO" "$PACKAGE_NAME" &> "$CHECKOUT_LOG"
done

# For now assume that if there is a DTK directory in the main LCM
# directory, it contains a DTK version that we can use for
# Trilinos.
if [ -e DataTransferKit ]; then
    cp -p -r DataTransferKit Trilinos
fi

# Clone wiki too to update info for latest known good commits.
if [ -d "Albany.wiki" ]; then
    rm "Albany.wiki" -rf
fi
git clone git@github.com:gahansen/Albany.wiki.git

./clean-config-build-test-mail-all.sh

cd "$LCM_DIR"
