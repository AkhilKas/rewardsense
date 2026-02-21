Project Scoping and Design Document 

RewardSense: A Cost-Aware, Explainable Credit Card Recommendation System 

Aditya Shenoy, Akhilesh Kasturi, Arjun Vinay Avadhani, Rahul Suresh, Vidya Kalyandurg 

 

1. Introduction 

Credit card reward programs have evolved into complex, multi-dimensional incentive systems. Modern credit cards offer rotating bonus categories, tiered reward structures, limited time merchant offers, statement credits, annual credits, and large welcome bonuses that require careful spending strategies to maximize value. While these rewards can provide substantial benefits, the cognitive and organizational burden placed on consumers is significant. Users must remember which card to use for each merchant, track quarterly caps, activate rotating categories, and understand how to optimally redeem points. 

This project proposes RewardSense, a real time credit card recommendation and optimization system that addresses this complexity through a combination of deterministic financial logic, machine learning personalization, and large language model (LLM) based explainability.  

The system further adapts to user behavior by learning personalized point valuations, detecting habitual spending patterns, and recommending new cards that improve the user’s overall reward efficiency. Each recommendation is accompanied by a transparent explanation describing the factors that influenced the decision. RewardSense targets financially active consumers who own multiple credit cards and want to maximize rewards without manual tracking.  

 

 

 

 

 

 

 

 

2. Dataset Information 

2.1 Dataset Overview and Purpose 

RewardSense relies on two complementary datasets: 

Credit Card Rewards Dataset 

Synthetic User’s credit cards owned Dataset 

Together, these datasets enable both accurate reward computation and realistic personalization modeling while avoiding privacy concerns associated with real financial data. The credit card dataset supports deterministic reward computation and eligibility logic, while the synthetic user dataset provides supervised and semi-supervised signals for learning personalized reward valuations and recommendation preferences. 

2.2 Data Card Summary 

Credit Card Dataset 

Size: ~100 cards 

Fields: reward rates, caps, fees, credits, expiration dates 

Purpose: reward computation and eligibility logic 

User Dataset 

Users: ~100 

Fields: user_id, card_id, redemption_preference 

2.3 Data Sources 

Issuer websites (Chase, American Express, Citi, etc.) 

Aggregator platforms (NerdWallet, The Points Guy) 

Scraping jobs are designed to respect rate limits and site terms of service. 

Link to Data Card - https://github.com/avadharj/rewardsense/blob/main/docs/data_card.md 

2.4 Data Rights, Privacy, and Governance 

With privacy in mind, we propose separating user identity from transaction details, encrypting data at rest and in transit, and allowing users to opt out or delete data. Our data pipeline will ensure that no PII beyond what is necessary is stored. We will also version and document our datasets thoroughly. This not only aids reproducibility but also governance and compliance: by versioning data and code, every model decision can be audited, supporting regulatory requirements. 

3. Data Planning and Splits 

Data Preprocessing: For the transactions data, preprocessing steps include: 

Cleaning: Ensuring all transactions have valid categories (we will maintain a mapping from merchant names to a standardized category code, possibly using Merchant Category Codes). Any missing or anomalous data (since it’s synthetic, anomalies are by design if any) will be handled or filtered. 

Feature Engineering: Computed features (expected reward or point value per card) could be used to train a model or as part of rule-based logic.  

Normalization: Ensure amounts and rewards are in comparable units (e.g., some cards give points which we convert to monetary value using the user’s personalized point valuation). 

Data Splitting: For model training and evaluation, we will split the user transactions dataset into training, validation, and test sets. A sensible strategy is a time-based split: use earlier transactions as training data and later transactions as validation/test, to mimic the model being deployed and tested on “future” data. This respects temporal order and avoids leakage of future behavior into training.  

Data Management: We will manage data through pipeline orchestration (Airflow or Vertex Pipelines). Each split (train/val/test) will be saved distinctly, so that model training uses the correct sets. We’ll ensure reproducibility by seeding any random processes in data generation. Through CI processes, we will include data validation tests. 

 

 

 

 

 

 

 

 

 

 

 

4. GitHub Repository Structure 

Here is a concise, technical summary that preserves the full folder structure while removing redundancy. This version is appropriate for a design doc or MLOps report. 

The project is hosted in a GitHub repository organized using industry-standard MLOps conventions to support reproducibility, scalability, and maintainability. 

README.md – Provides an overview of the project, setup instructions, usage examples, and a summary of the repository structure. 

data/ – Contains dataset documentation (e.g., data_card.md) and placeholders for data artifacts. Large or sensitive datasets are not committed directly and may be managed via external storage or DVC. 

notebooks/ – Jupyter notebooks for exploratory analysis and prototyping. All experimental work is later refactored into production pipelines. 

src/ – Core application and ML logic: 

src/data_pipeline/ – Data ingestion, web scraping, preprocessing, and synthetic data generation. 

src/models/ – Model training and evaluation logic, including recommendation and point-valuation components. 

src/app/ – Inference and serving layer (e.g., Flask/FastAPI API for real-time recommendations). 

src/utils/ – Shared utilities for logging, configuration, and reward calculations. 

models/ – Placeholders or metadata for trained model artifacts, with actual binaries stored in a model registry or cloud storage. 

infrastructure/ – Deployment and orchestration configuration, including Dockerfiles, Kubernetes manifests, Airflow DAGs, and CI/CD resources. 

tests/ – Unit and integration tests for data pipelines, models, and API endpoints, executed automatically in CI. 

docs/ – Design documentation, architecture diagrams, and project reports. 

 

Link to Github Repository - https://github.com/avadharj/rewardsense 
 

 

 

5. Project Scope 

5.1 Problems Addressed 

The project addresses several key problems in the domain of credit card reward optimization and ML lifecycle management: 

Which Card to Use? (Real-time Decision) – Consumers with multiple credit cards often struggle to know which card will maximize rewards for a given purchase. Different cards have different reward rates depending on the category or merchant, and those can change over time (e.g., quarterly 5% categories).  

Benefit Tracking Complexity – Credit card rewards aren’t static; many cards have rotating bonus categories, caps on high-reward spending, limited-time merchant offers, or annual benefits (like $10 monthly dining credit). Tracking all these conditions is cumbersome for a user.  

Net Value Optimization – Not all rewards are “free” – cards have annual fees. The naive approach of maximizing points on a single transaction might not yield the best net value when considering fees or long-term goals.  

Personalized Points Valuation & Redemption – The value of a reward point or mile can vary wildly depending on how it’s redeemed. The problem here is to learn each user’s effective valuation of different point currencies and redemption habits, so that recommendations and advice can be tailored.  

Habitual Spend Detection – Many people have recurring purchases (morning coffee, weekly groceries) and they might default to one card out of habit, which might not be optimal. Conversely, if one card is consistently best for a certain merchant, the user could save time by always using it. The problem is recognizing these patterns to either 1) automate the recommendation (set a default card for that merchant to streamline future decisions), or 2) alert the user if their habit is suboptimal (“You often use Card A at Starbucks, but Card B would earn more”). 

Portfolio Gaps (“What Card Next?”) – Given the multitude of credit cards available, users often wonder if they have the right mix. They may be missing out on rewards in a category they spend a lot on or could benefit from a premium card perks. Identifying which new credit card would most improve a user’s rewards given their profile is a problem we tackle.  

Explainability and Trust – Any AI/ML recommendation system faces the issue of user trust. Users want to know why a certain card is recommended.  

 

5.2 Existing Solutions and Limitations 

There are a few existing solutions and approaches that partially address these problems, though each has limitations that our project aims to overcome: 

Manual Strategies and Static Guides: Many savvy users maintain their own spreadsheets or use static online guides for credit card rewards. Websites like NerdWallet or The Points Guy publish recommended cards by category, but these are generic and not tailored to an individual’s wallet.  

WalletFlo and Others: WalletFlo is another solution focusing on managing credit card perks, tracking welcome bonuses, and alerting users to changing bonus categories. It emphasizes not leaving value on the table, like our goal. It includes features like showing which card to use for the most points and tracking progress on bonuses. One limitation is that it does not optimize for net value (annual fees) explicitly, and it uses rule-based calculations rather than predictive modeling.  

Credit Card Recommendation Engines: Some financial services or forums offer advice on “what card to get next” through static calculators or community input. These are usually one-time analyses and do not consider the user’s actual transaction history in depth.  

 

 

 

 

 

 

 

 

 

 

 

 

 

5.3 Proposed Solution Architecture 

 

 

RewardSense combines: 

Deterministic financial logic for correctness  

ML models for personalization 

LLMs with RAG for explanations and advice 

Automated MLOps pipelines for continuous improvement 

 

Data Layer 

External Data Sources: Credit card issuer websites, aggregator APIs (Credit Card Bonuses API), and user transaction inputs feed into the system. The Scraping Service runs scheduled jobs via Airflow DAGs, extracting reward structures, promotional offers, and benefit details. Data is validated, cleaned, and loaded into PostgreSQL for persistent storage and Redis for real-time feature serving. 

Processing Layer 

The Feature Engineering Pipeline transforms raw data into ML-ready features: merchant-to-category mapping, reward rate normalization, cap utilization tracking, and bonus progress calculation. User profiles are enriched with spending patterns detected via DBSCAN clustering and historical redemption behavior. All features are versioned and stored in the Feature Store for both training and inference consistency. 

Model Layer 

The hybrid recommendation engine combines: (1) Deterministic Reward Calculator for precise reward computation using current card data, caps, and multipliers; (2) XGBoost Points Valuation Model for personalized redemption value prediction; (3) PuLP-based Portfolio Optimizer for next-card recommendations; (4) GPT-4o Mini for generating natural language explanations. MLflow tracks all experiments and manages model versioning in the Model Registry. 

Serving Layer 

FastAPI serves the recommendation endpoints, deployed as containerized microservices on GKE. The API Gateway (Kong/Nginx) handles routing, rate limiting, and authentication. Key endpoints include: /recommend (real-time best card), /track (benefit status), /next-card (portfolio optimization), and /explain (detailed reasoning). Redis caching ensures sub-500ms latency for frequently accessed card data and user profiles. 

CI/CD Pipeline 

GitHub Actions automates the entire deployment lifecycle: code linting (Ruff/Black), unit and integration testing (pytest), Docker image building, security scanning (Trivy), and deployment to GKE. The pipeline includes staging environment validation before production rollout, with automated rollback capabilities on failure detection. 

Monitoring Layer 

Comprehensive observability through: Prometheus collecting system and application metrics; Grafana dashboards for visualization; Evidently AI for ML-specific monitoring (data drift, prediction drift, model performance); GCP Cloud Logging for centralized log aggregation; PagerDuty/Slack integration for alerting. Automated retraining triggers when drift exceeds configured thresholds.  

 

 

6. Workflow and Bottleneck Analysis 

Bottleneck Detection: In analyzing the above workflow, we identify potential bottlenecks and challenges: 

Data Freshness Bottleneck: The credit card data scraping could become a bottleneck if not managed – for example, if a website significantly changes layout, the scraper might break and fail to update our database. This could lead to outdated recommendations. To mitigate, we plan redundancy (multiple sources) and quick alerts/tests on the scraping job.  

Real-Time Inference Speed: When a recommendation request comes in, the service must aggregate user data + card data + compute rewards quickly. If our logic or model is too complex (e.g., a very large ML model or needing to query a large database each time), it could introduce latency. Caching can alleviate this (like the user’s points valuation or remaining caps) in memory.  

Bottlenecks in CI/CD: The current approach automates a lot, which is good for speed but introduces points of failure. We will optimize CI by running tests in parallel and using cloud build agents.  

Integration Bottlenecks: Integrating many moving parts (scraper, pipeline, model, API, monitoring) means the system is only as fast as its slowest component. For example, if monitoring is not real-time enough, we might not catch a failing scraper quickly, causing stale data bottleneck above.  

 

 

 

 

 

 

 

 

 

 

7. Metrics and Objectives 

Technical Performance Metrics: 

Reward Gain (Value Added): A key metric is how much additional reward value a user gets by following our recommendations versus their baseline behavior.  

Latency: The time from a user request to recommendation response.  

Throughput and Scalability: The system should handle concurrent users.  

Model Metrics 

Uptime and Reliability 

Objectives: 

Maximize User Rewards Net Value 

Personalize User Experience 

Ensure Model Reproducibility & Automation 

Business Goals and Alignment: 

User Retention and Growth: In a product context, providing clear value (more rewards with less effort) will drive users to continue using the app and tell others. A business metric might be Monthly Active Users (MAU) or retention rate.  

Compliance and Risk Mitigation: Objectives here include passing any required compliance checks (for instance, if handling financial data, being PCI compliant). While not directly measured in numbers, success criteria like “All user data is encrypted, and we have passed a security audit” are important.  

 

 

 

 

 

 

 

 

8. Failure Analysis 

8.1 Data Pipeline Failures: 

Data Quality Issues: Even if scraping works, data might be erroneous (e.g., a parsing error yields a 50% reward instead of 5%). If such outlier data goes unnoticed, the model would give a wrong recommendation. Mitigation: Use data validation checks after scraping. Also, test the data through the recommendation logic: if one card suddenly scores dramatically higher than historically, flag it for review. 

Synthetic Data Assumptions: Since our model is trained on synthetic data, there’s a risk the model learns patterns that don’t hold in real life (simulation bias). This could cause model failures when exposed to real user behavior. Mitigation: Make synthetic data as diverse and realistic as possible. Post-deployment, gather real usage data and retrain or fine-tune models to reality.  

 

8.2 Model & Algorithm Failures: 

Model Drift: Over time, user behavior or card offerings may drift away from what the model was trained on. For example, if travel nearly stops (like early 2020) and people spend more on groceries, a model of heavily weighting travel rewards might misfire. Or if a new type of reward (say crypto rewards) becomes popular and wasn’t in training data. Mitigation: Implement continuous monitoring of model performance. We’ll track if users start deviating from recommendations or if our “accuracy” on recent data falls. Using tools like Evidently AI to monitor data and prediction distributions can automatically highlight drift. When a drift is detected beyond a threshold, trigger retraining (Continuous Training). Keep a model registry with versioning so we can roll back to a previous model if a new one performs worse. 

Biases or Wrong Optimization Objective: The model might over-optimize certain metrics but not actual user happiness. For instance, it might push too hard to complete a signup bonus (because that gives a big reward) even if it inconveniences the user or goes against their preference. This could cause user dissatisfaction – a form of failure. Mitigation: Incorporate user preferences properly into the objective function (if a user hates using a certain card, perhaps mark it or weigh it down). Also include user feedback loops: if a user consistently ignores a certain recommendation, the system should learn to adjust (maybe they have a reason, like they prefer a lower reward on a simpler card – some users do value simplicity). 

 

8.3 Deployment & Infrastructure Failures: 

CI/CD Pipeline Failures: The automated pipeline might fail to deploy (e.g., a Docker build error, a test failing). This could block updates or, worse, if not noticed, leave the system outdated. Mitigation: Use notifications on CI/CD (so any failure pings the team). Also, ensure that a failing update doesn’t take down the current service – use blue/green deployments or only switch traffic when new instance is healthy.  

Security Breaches: Handling financial recommendations means we might deal with sensitive info (though we’re avoiding actual card numbers or personal data, but still we have spending data). A breach or leak would be a major failure. Mitigation: Implement strong security measures: encrypt data at rest, and secure API endpoints. Conduct code reviews for security, use dependency scanning to avoid vulnerabilities in libraries.  

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

 

9. Deployment Infrastructure (GCP) 

Cloud Environment & Compute: Containerized application deployed on Google Kubernetes Engine (GKE) for high availability and auto-scaling. Each component (scraping job, training pipeline, API server) runs in isolated pods. Kubernetes ensures cloud-agnostic portability across GCP, AWS, or Azure. 

Containerization: All services packaged as Docker images with consistent environments and dependencies. Images stored in GCP Artifact Registry, versioned by git commit tags. 

CI/CD Pipeline: 

CI: GitHub Actions runs automated tests and builds Docker images on code push 

CD: Jenkins/Cloud Build deploys images to Kubernetes cluster via manifests or Helm charts 

Orchestration: Apache Airflow schedules ML pipeline jobs and batch processes 

Automated quality gates ensure code and models pass tests before deployment 

Data Storage: 

BigQuery: Data warehouse for transactions and analytics (next-card recommendations, "what if" scenarios) 

Cloud SQL (PostgreSQL)/Firestore: Low-latency serving for user profiles, card lists, and bonus trackers 

Cloud Storage (GCS): Artifact storage for model files, datasets, and static assets 

Vertex AI Feature Store (Optional): Manages features for training/serving consistency 

Architecture Summary: Cloud-based containerized microservice architecture leveraging GCP managed services (GKE, BigQuery, Vertex AI) for scalability and portability. Secure network connections with automated data flow from collection through training, serving, and monitoring. 

 

 

 

 

 

10. Monitoring Plan 

Data Quality and Freshness: 

Track scrape job execution (expected: ≥X cards updated weekly, job run within 24h) 

Monitor dataset statistics (reward rate distributions, category counts) for anomalies 

Validate transaction data (no negative amounts, required fields present, ingestion lag) 

Alert on pipeline failures or unexpected data shifts 

Model Performance: 

Log recommendations and track "% of optimal reward achieved" metric 

Monitor input distribution drift vs. training data using Evidently 

Track prediction consistency and detect anomalies in point valuations 

A/B test new model variants when applicable 

Application and API: 

Monitor latency (p90, p99 ≤ 1s target), throughput, and error rates via Prometheus/Stackdriver 

Track resource utilization (CPU, memory, database load) for capacity planning 

Implement heartbeat checks for uptime monitoring (Pingdom/Stackdriver) 

Alert on: high error rates (>5% in 5 min), resource exhaustion (>90% memory), API downtime 

Business Metrics: 

User engagement trends (recommendations per user/week) 

Aggregate additional rewards earned through suggestions 

Next-card recommendation acceptance rates 

Alerts and Response: 

Critical alerts: API down, scraper failure (1 day), high error rate (>5%), model drift threshold exceeded 

Notification channels: Email/Slack for immediate response 

Logging: Centralized in Cloud Logging with appropriate detail levels (info for decisions, error with stack traces) 

Documentation: Runbooks for common issues with step-by-step resolution procedures 

Alerts trigger automated retraining, rollback, or investigation workflows. 

11. Success and Acceptance Criteria 

Correctness & Core Functionality: System must accurately recommend optimal cards in real-time. Criterion: 95% of test cases match expected optimal card across all categories and user profiles.  

Explainability: All recommendations must include coherent, factually correct explanations. Criterion: Explanations accurately reference actual card benefits (e.g., verified 3x dining rates).  

User Value: Demonstrable improvement in rewards earnings. Criterion: 10-20% net rewards improvement over baseline strategies in simulations; positive user feedback on advice quality.  

Performance and Scalability:  

API latency ≤ 1 second (95th percentile)  

Handle 50+ concurrent users without errors  

Tested via load testing tools (JMeter/Locust)  

Robustness and Reliability:  

Graceful handling of edge cases (unknown categories, invalid inputs)  

99%+ uptime during testing period  

Zero-downtime deployments via rolling updates  

MLOps Process Adherence:  

Reproducibility: Independent reproduction of model results from repository with same data/metrics  

Automation: Complete CI/CD from commit to deployment without manual intervention  

Containerization: Docker images and Kubernetes configs demonstrable; pods visible via kubectl; easy replica scaling  

Monitoring & Alerting: Functional dashboards with demonstrated alert triggers (e.g., scraper failure alerts)  

Documentation & Maintainability: Comprehensive README, code comments, data dictionary, and pipeline diagrams enabling new developers to understand, run, and extend the system.  

Compliance & Security: No sensitive data in public repos, secrets managed securely, passes static analysis security checks.  

User Acceptance: Positive feedback from test users; stakeholder scenario tests handled correctly per requirements. 

 

12. Timeline Planning 

 

 

 

 

 

 

13. Future Work 

Incorporating credit score and approval odds into the “what card next” – a truly robust recommendation would consider if the user can likely get the card (no point recommending an exclusive card if the user’s credit is average). We assume for now user can get most mainstream cards. 

Access to user’s transactions for better judgement regarding ‘what card to choose’  

 

14. Conclusion 

RewardSense demonstrates how MLOps principles can be applied to a realistic, financially grounded optimization problem. The project delivers a scalable, explainable, and cost-aware recommendation system while showcasing modern ML deployment practices in a cloud-native environment. 