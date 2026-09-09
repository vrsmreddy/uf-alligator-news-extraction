# UF Alligator News NLP Toolkit

An NLP pipeline for extracting **named entities** and **coreference chains** from news articles using transformer-based models.

## Overview

This project applies state-of-the-art NLP techniques to analyze news articles from the UF Alligator, demonstrating how unstructured text can be transformed into structured analytical insights. The pipeline is directly applicable to financial news analytics, credit risk signal extraction, and entity-level event tracking.

## Features

- **Named Entity Recognition (NER)** — Extracts organizations, people, locations, dates, and other entities using SpaCy's transformer-based model (`en_core_web_trf`)
- **Coreference Resolution** — Identifies and links pronoun references to their named entities using the Coreferee library
- **Transformer-based accuracy** — Leverages `en_core_web_trf` for state-of-the-art NLP performance

## Tech Stack

- **Python**
- **spaCy** (`en_core_web_trf`) — Transformer-based NLP model
- **Coreferee** — Coreference resolution pipeline
- **Hugging Face Transformers** (via spaCy backend)

## Installation

```bash
pip install spacy coreferee
python -m spacy download en_core_web_trf
python -m coreferee install en
