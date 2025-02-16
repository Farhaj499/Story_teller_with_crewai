import yaml
from crewai import Agent, Task, Crew, Process
from crewai.project import CrewBase, agent, task, crew
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = os.getenv("MODEL")
api_key = os.getenv("GEMINI_API_KEY")

llm_instance = ChatGoogleGenerativeAI(model=model, api_key=api_key)

agents_config = 'config/agents.yaml'
tasks_config = 'config/tasks.yaml'

@CrewBase
class StoryCrew:
    
    @agent
    def story_planner_agent(self)-> Agent:
        config = self.agents_config['story_planner_agent']
        return Agent(
            role = config['role'],
            goal = config['goal'],
            backstory = config['backstory'],
            verbose = config.get("verbose", False),
            llm_instance = llm_instance
        )
    @agent
    def story_writer_agent(self)-> Agent:
        config = self.agents_config['story_writer_agent']
        return Agent(
            role = config['role'],
            goal = config['goal'],
            backstory = config['backstory'],
            verbose = config.get("verbose", False),
            llm_instance = llm_instance
        )
    @agent
    def story_summarizer_agent(self)-> Agent:
        config = self.agents_config['story_summarizer_agent']
        return Agent(
            role = config['role'],
            goal = config['goal'],
            backstory = config['backstory'],
            verbose = config.get("verbose", False),
            llm_instance = llm_instance
        )
        
    @task
    def outline_task(self)-> Task:
        config = self.tasks_config['outline_task']
        return Task(
            description = config['description'],
            expected_output = config['expected_output'],
            agent = self.story_planner_agent()
        )
    @task
    def write_task(self)-> Task:
        config = self.tasks_config['write_task']
        return Task(
            description = config['description'],
            expected_output = config['expected_output'],
            agent = self.story_writer_agent()
        )
    @task
    def summarize_task(self)-> Task:
        config = self.tasks_config['summarize_task']
        return Task(
            description = config['description'],
            expected_output = config['expected_output'],
            agent = self.story_summarizer_agent()
        )
    
    @crew
    def crew(self)-> Crew:
        return Crew(
            agents = [
                self.story_planner_agent(),
                self.story_writer_agent(),
                self.story_summarizer_agent()
            ],
            tasks = [
                self.outline_task(),
                self.write_task(),
                self.summarize_task()
            ],
            process = Process.sequential,
            verbose = True
        )
    
    