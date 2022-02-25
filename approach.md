# Our Approach

## Introduction
Advances in brain recordings present an opportunity to improve our understanding of brain function in unprecedented detail. The science of interpreting brain signals has myriad applications from the biology of neurological disease to interfacing brain and machine. Latent variable models (LVMs) are a promising tool for interpreting increasingly complex neural data. The Neural Latents Benchmark (NLB) challenge was created to coordinate efforts toward more advanced LVMs. The first NLB challenge occurred from September 2021 until January 7th, 2022. The challenge was to predict neuron activity spanning four tasks and brain areas across seven datasets. Our team from AE Studio was pleased to be announced as the winners of the first NLB challenge. Here we describe the approach we took and how we arrived at our solution.

We believe that our winning solution is the result of a system of behaviors and best practices for software engineering and data science. By employing these behaviors and practices to adapt existing state-of-the-art LVMs, we were successful in submitting the top performing model across all datasets in the NLB challenge. We believe these behaviors are important to enumerate because we have found they accelerate the pace of development and produce robust and reusable code. We hope that neuroscience researchers will adopt these best practices and that the community at large will benefit from research which is sharable and easily reproducible.

## Behaviors and Practices

Agile software development argues that a project presents three degrees of freedom: scope, time, and resources. Fixing all three simultaneously over-constrains the problem and there is no solution: the project will either be late, exceed its budget, or fail to deliver the promised scope. Therefore, a successful project must present at least one degree of freedom. For this project, time and resources were fixed, thus requiring a flexible scope.

We concluded that we would not have time to ideate, implement, and experiment with a novel machine learning architecture. We needed a plan that would achieve competitive results as quickly as possible while leaving room for iterative improvement. We decided instead to adapt the currently available state-of-the-art models for neural population dynamics (i.e. Neural Data Transformers[^ndt] and AutoLFADS[^autolfads]) and make incremental improvements within the NLB challenge’s timeframe. By enumerating a backlog[^backlog] of small, incremental improvements and implementing them sequentially, we ensured that potential submissions were available at each step of the way (as opposed to designing an optimal implementation and running out of time before all finalized components could be assembled).

[^ndt]: https://doi.org/10.51628/001c.27358
[^autolfads]: https://doi.org/10.1101/2021.01.13.426570
[^backlog]: The backlog was ordered based on estimates of implementation time and the expected value of result improvement and was adjusted over time based on new information from experiments.

### Agile Software Development

This method of implementing small incremental changes, measuring results, and adjusting the plan in light of these results is at the core of agile software development. If time had allowed implementing a novel machine learning model, we would simply have adopted the same approach with a more ambitious scope. 

Code quality is similarly important to agile software development. Reliable and easily-modifiable code allows us to build atop previous work in order to accelerate subsequent experiments. We achieve this by following clean code principles, automated testing, and clear documentation. Ideally, following these practices ensures that, at every step, we have deliverable software.

Communication among the team is also critical for moving quickly together. We used the agile practices of week-long iterations with weekly demo and planning meetings, daily standups (implemented asynchronously via slack), and frequent 1-on-1 pairings. We also used Pull Request reviews on each new experiment result, helping everyone on the team remain up to date with new results and ensuring we could build on top of each other’s work.

### Data Science

When developing machine learning solutions for a product, best practices[^rules_of_ml] suggest that the first and most important step is to build a solid data pipeline. Next, a simple model or even a heuristic can be created to test the machine learning framework end-to-end. With a fully tested pipeline, iterative model development becomes easy and efficient.

This was our exact approach during development for the NLB 2021 challenge. First, we assembled data pipelines and simple models and then, gradually explored more complex alternatives. Our data pipeline allowed us to easily build “adapters” to convert data to any format that specific models required, which was especially useful when we adapted state-of-the-art model codebases.

We aimed for small experiments in order to share results frequently, only making one type of change at a time in order to isolate the effect. Clearly separating engineering tasks (such as writing utility functions and tests) from research tasks (such as experiments making changes to models and measuring their performance) helps to better focus work and decrease implementation times.

We also relied upon a strong understanding of fundamental data science concepts in order to identify which techniques were most likely to improve modeling results. In representation learning, there are three main design choices: architecture, regularization, and optimization. These choices are not mutually exclusive. For example, batch normalization[^batch_normalization] is an architectural decision that is thought to create a smoother gradients, aiding optimization[^optimization]. Ensembling techniques are often used in competitions to boost performance and can be seen as a form of regularization (although ensembling in deep neural networks is not well understood[^ensemble_not_understood]). We identified ensemble averaging[^ensemble_averaging] and Bayesian hyperparameter optimization as methods that offered the best trade-off between fast implementation and potential performance improvements. 

In addition, we heavily relied upon cloud environments for flexibility and minimizing costs. We worked almost exclusively within AWS on EC2 instances, and would frequently change our instance type from low-power and cheap instances for software development and simple data processing or analysis (eg. r5.large, which costs ~$1 per 9-hour workday), to high power multi-GPU instances for intensive training (eg. p3.8xlarge with 4 GPUs). We also used S3 extensively for sharing results and storing artifacts from training runs.

[^rules_of_ml]: https://developers.google.com/machine-learning/guides/rules-of-ml
[^batch_normalization]: https://arxiv.org/abs/1502.03167
[^optimization]: https://arxiv.org/abs/1805.11604
[^ensemble_not_understood]: https://arxiv.org/abs/2012.09816
[^ensemble_averaging]: https://en.wikipedia.org/wiki/Ensemble_averaging_(machine_learning)

## Methods

For our winning model submission, we trained 100+ neural data transformer models[^our_fork] (per dataset) through Bayesian hyperparameter optimization. A subset of the available model checkpoints were then ensembled using ensemble averaging.

Hyperparameter optimization was executed via Ray Tune[^ray_tune] with the scikit-optimize implementation of Bayesian optimization[^ray_tune_bayesian]. We trained these models on an AWS p3.8xlarge[^p3_instance] instance[^training_times]. From each trial, one model snapshot was saved for the epoch with the best masked loss, and another for the epoch with the best unmasked loss. Once all trials were complete, the validation co-bps[^co_bps] was measured for each checkpoint.

Ensembles were formed for each dataset by averaging the output unit firing rates of the top N checkpoints (when ordered by decreasing validation co-bps). The number of models in each ensemble was chosen independently for each dataset in order to maximize the validation co-bps of the ensemble predictions[^ensemble_sizes]. We examined the effects of different ensemble sizes and found that the co-bps remained relatively stable around the local maximum as we varied the number of ensembled models, suggesting that our choice of ensemble sizes were reliable and that the test co-bps would not be highly sensitive to ensemble size.

[^our_fork]: Using our forked version of the [original neural data transformer codebase](https://github.com/snel-repo/neural-data-transformers)
[^ray_tune]: https://docs.ray.io/en/latest/tune/index.html
[^ray_tune_bayesian]: https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#skopt
[^p3_instance]: https://aws.amazon.com/ec2/instance-types/p3/
[^training_times]: For training times with this setup, see the second table in our codebase [README](https://github.com/agencyenterprise/ae-nlb-2021/blob/master/README.md#model-description)
[^co_bps]: https://eval.ai/web/challenges/challenge-page/1256/evaluation
[^ensemble_sizes]: For details on ensemble size, see the first table in our codebase [README](https://github.com/agencyenterprise/ae-nlb-2021/blob/master/README.md#model-description)


## What did not work

We also tested potential improvements that did not yield the desired results. We think it is useful to the broader community to detail these paths as well.

### Stacking

We tried stacking our set of NDT models (e.g. lasso regression on the outputs of the individual models) with various forms of stacking models and setups, but this did not outperform a simple averaging of these models. Note that our winning ensemble was in the space of solutions for the stacking models but these solutions were not found. We hypothesize that we may not have regularized our stacking models optimally. Time constraints did not allow a complete exploration of stacking and future work can explore stacking as a method to improve performance.

### Per Neuron Ensembling

We attempted to create an ensemble by picking the top classifiers, per neuron, and using the average of those rates for the corresponding neuron. Perhaps not surprisingly, this resulted in a significant amount of overfitting. Future work can investigate regularization strategies to prevent this overfitting. Another open question is whether or not these models which have overfit to a subset of neurons generalize to others. 

### AutoLFADS/LFADS

We were unsuccessful in quickly adapting the currently available open-source implementation of AutoLFADS[^autolfads_opensource], which is tightly tied into Google cloud and does not yet include the architectural changes for use with held-out neurons. Instead we adapted an alternative open-source LFADS[^hierarchical_lfads] model for use with the masking and hyper-parameter optimization from NDT. This allowed for our LFADS work to benefit from work done in parallel on NDT while giving us the opportunity to experiment with population based training and a coordinated masking strategy. Nevertheless, our final LFADS implementation was not competitive with either AutoLFADS or our own optimized NDT. Our LFADS ensemble models’ performances lagged our NDT-only ensembles, and a mixed LFADS and NDT ensemble was competitive with other leaderboard results but ultimately did not achieve the highest scores.

[^autolfads_opensource]: https://snel-repo.github.io/autolfads/
[^hierarchical_lfads]: https://github.com/lyprince/hierarchical_lfads

## Conclusion

Through a set of behaviors and best practices we were quickly able to leverage existing state-of-the-art research, namely neural data transformers, and tune these models to a winning solution. It is well-known that ensembling is a reliable technique to boost model performance, but exactly why it works in the case of predicting latent factors has the potential to reveal novel insight through further research. By setting a new best in co-bps across all NLB challenge datasets, we have achieved a new benchmark that researchers can use to help understand the limits of what is possible in predicting latent factors. 
