# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Development Rules
1. Do not commit / push code on your own.
2. When running scripts, use 'conda activate pm_touch', except for scripts living inside @real_data_transfer: here use 'conda activate pm_real' instead.
3. When training networks, don't use wandb offline. Always log them online. Ask the user again if prompted for login.

## Reporting Visualization Results, Implementation Details, Quantitative Results
1. Publish findings in html, inside log/
2. Inside the reports, don't make up or use complicated technical jargon. Instead, write things out so that it is easy to understand.
3. Always explain abbreviations with a parenthesis if you need to use them.

