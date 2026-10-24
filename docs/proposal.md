# CS5284 Group Project Plan

*Predicting Monthly Road-Segment Accident Counts with Graph Neural Networks: Richer Road Features, Accident History and an Interactive Risk Map*

**Group ID:** 01

**Team members:** Aaron Teo Rui Kang, Cheong Fung Wei, Liu Jing Jing, Duggal Vidurveer, Jason Ciu Putra Sung

## Project motivation

Thousands of road accidents happen every month in each US state, yet it is hard to say in advance which roads are most at risk. Road-level forecasts could help agencies decide where to focus inspections and safety measures. A road network is naturally a graph (intersections joined by road segments), and neighbouring roads share traffic, layout and risk, so graph machine learning is a natural tool. Accidents are rare events on any single road, which makes the problem hard and makes it important to understand *why* a model works or fails.

## Project description

We will predict how many accidents each road segment will have in a future month. We build on the public benchmark of Nippani et al. (NeurIPS 2023 Datasets and Benchmarks): about nine million official accident records from eight US states, each linked to a road graph built from OpenStreetMap (nodes are intersections, edges are road segments) with monthly accident counts, weather and, for a few states, traffic volume. The original models use the network structure, weather and only basic road information (length and a road-type flag). As far as we can tell from their paper and code, past accidents are used only as training labels and never as inputs. Evaluation is chronological: train on earlier years, test on later years.

## Proposed solution

1. **Reproduce the baselines.** Re-implement the paper's no-graph MLP and GraphSAGE models on one or two smaller states (candidates: Delaware, Maryland; final choice after a data audit in Week 8), with the same splits and metrics (MAE, AUROC), and compare against the published numbers.
2. **Initiative 1a: new road features.** Build a pipeline that extracts richer attributes from OpenStreetMap (speed limit, lanes, cycle lanes, traffic signals, junction type, lighting), matches them to segments, and measures with feature-group ablations how much each adds.
3. **Initiative 1b: accident history.** Add past-accident counts for each road and its neighbouring roads, built only from earlier periods so the model never sees the answer, and test with a walk-forward evaluation. Whether to use a fixed window of earlier years or a rolling window of recent months will be decided after exploring the data.
4. **Initiative 2: model improvements.** Try changes such as a model that starts from a history-based estimate and learns a correction, a training loss and sampling that reflect that most roads have no accidents, and (if time allows) letting each road weigh its neighbours differently. Each change is compared with the previous step over several random seeds, using MAE, AUROC and precision/recall on the highest-risk roads, and we explain why it helps or does not.
5. **Initiative 3: interactive visualisation.** An interactive map (e.g. Bokeh/Plotly) where users explore predicted accident risk road by road and month by month and compare it with what actually happened.
6. **Initiative 4:** explore the relevance of the model built on US roads to Singapore roads.



## Project milestones


| Week                 | Milestone                                                                                      |
| -------------------- | ---------------------------------------------------------------------------------------------- |
| Wk 7 (to Oct 4)      | Submit plan and signed team contract; start data audit of candidate states.                    |
| Wk 8 (Oct 5-11)      | Finish audit and select state(s); explore and visualise the data; start reproducing baselines. |
| Wk 9 (Oct 12-18)     | Complete reproduction versus the paper; build the OpenStreetMap extraction pipeline.           |
| Wk 10 (Oct 19-25)    | Add OpenStreetMap features; run feature-group ablations; design history features.              |
| Wk 11 (Oct 26-Nov 1) | History-feature experiments (walk-forward, no leakage); Zoom meeting with TA.                  |
| Wk 12 (Nov 2-8)      | Model changes and multi-seed comparisons; analysis of what helps and why.                      |
| Wk 13 (Nov 9-15)     | Interactive map; draft report, slides and notebook cleanup.                                    |
| Wk 14 (Nov 16-22)    | Final submissions, due Sun Nov 22, 11:59 pm.                                                   |




## References

[1] A. Nippani, D. Li, H. Ju, H. N. Koutsopoulos, H. R. Zhang. *Graph Neural Networks for Road Safety Modeling: Datasets and Evaluations for Accident Analysis.* NeurIPS Datasets and Benchmarks, 2023. [https://arxiv.org/pdf/2311.00164](https://arxiv.org/pdf/2311.00164)

[2] Code: [https://github.com/VirtuosoResearch/ML4RoadSafety](https://github.com/VirtuosoResearch/ML4RoadSafety). Data: [https://doi.org/10.7910/DVN/V71K5R](https://doi.org/10.7910/DVN/V71K5R)

# CS5284 Group Project: Team Contract

*Group ID: 01. Project: Predicting Monthly Road-Segment Accident Counts with Graph Neural Networks*

## Tasks assigned to each team member


| Team member          | Student no. | Role                        | Coding tasks                                                                                                                                                    | Report and presentation                           |
| -------------------- | ----------- | --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------- |
| Aaron Teo Rui Kang   | A0155714E   | A: Data and map             | Data loading and audit; state selection; exploration plots; interactive map of predicted versus actual risk, fed by the shared predictions file.                | Data and exploration; interactive map             |
| Cheong Fung Wei      | A0229372B   | B: Baselines and evaluation | From-scratch MLP and GraphSAGE; training loop; metrics (MAE, AUROC, PR-AUC, precision/recall on top-risk roads); multi-seed runner.                             | Reproduction against the paper; evaluation method |
| Duggal Vidurveer     | A0357619N   | C: OpenStreetMap features   | Extraction pipeline (speed limit, lanes, cycle lanes, traffic signals, junction type); matching to road segments; feature-group ablations.                      | New road features and their effect                |
| Liu Jing Jing        | A034643J    | D: Accident history         | Leakage-safe history features for each road and its neighbours; fixed versus rolling windows; walk-forward test code.                                           | History features and the window decision          |
| Jason Ciu Putra Sung | A0219734Y   | E: Model improvements       | Model variants (history-based start with learned correction; loss and sampling for rare accidents; optional attention over neighbours); comparison experiments. | Model changes and why they help or not            |




## Signatures

By signing, each teammate agrees to the tasks assigned above.


| Name                 | Signature | Date       |
| -------------------- | --------- | ---------- |
| Aaron Teo Rui Kang   |           | 03/10/2026 |
| Cheong Fung Wei      |           | 03/10/2026 |
| Liu Jing Jing        |           | 03/10/2026 |
| Duggal Vidurveer     |           | 03/10/2026 |
| Jason Ciu Putra Sung |           | 03/10/2026 |


