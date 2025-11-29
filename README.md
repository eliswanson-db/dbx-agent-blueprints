dbx-agent-blueprints

### Example agents that solve particular architecture problems

1. Setting up some vector indexes  

1. agent with parallel unstructured retrieval workers    
    a. A "Good" variant defined as `Agent in Code`    
    b. A "Bad" variant defined as `Agent in Code`   
    **c. Simulate a model/agent development process by logging and UC registering "updated" model/agent**
     - `02c_lifesciences_agent_devs(Workflow with logNregister)`  

1. agent with unstructured retrieval, UC functions for simple queries, and pre-Genie context fetching

1. (Original) Stubbed out evaluation and promotion agent. This is not doing real evals currently, but mocking them and then promoting the most recent version to champion alias   
   **(New) notebook runs `mlflow.genai.evals` on the agent/model to see if it passes the selected threshold criterion for promoting model version as 'champion'.**
    - `04_model_eval_promote(Workflow with taskValues specification)`
    - we pass critical values to downstream task   
      
1. **Deploy model with champion alias to model serving endpoint** 
    - retrieve critical key values from upstream task that will determine if serving endpoint will be deployed/updated  
    - `05_deploy_agent(Workflow with taskKeys retrieval)`

1. **We create a workflow with the main tasks, which can be specified as a job in Databricks Asset Bundles**
    - `02c_lifesciences_agent_devs(Workflow with logNregister)`
    - `04_model_eval_promote(Workflow with taskValues specification)`
    - `05_deploy_agent(Workflow with taskKeys retrieval)` 

     