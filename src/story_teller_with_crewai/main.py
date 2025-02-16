from story_teller_with_crewai.crew import StoryCrew

def main():
    input_data = input("Describe your story: ")
    inputs = {
    "prompt": input_data,
}
    
    result = StoryCrew().crew().kickoff(inputs=inputs)
    print(result)
    