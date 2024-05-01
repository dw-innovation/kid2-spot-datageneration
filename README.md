# KID2 Spot Application

## Introduction

This is a preliminary readme, more information will follow

## Contents of this repository

This repository contains scripts related to the generation of a training database 
for the Spot Application.

The general idea of the pipeline is the following:
1) Bundle similar tags, assign natural language descriptors to create better semantic connections between language and the OSM tagging system
2) Generate random artificial queries including area definition, objects (incl. tags and descriptors) and relations/distances
3) Call GPT API to generate artificial natural sentences from the draft
