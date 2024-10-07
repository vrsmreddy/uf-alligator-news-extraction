import spacy
import coreferee

# Load the transformer-based SpaCy model and add Coreferee to the pipeline
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('coreferee')

# Article text
article_text = """
As Hurricane Helene swept through Gainesville Sept. 26, the Humane Society of North Central Florida lost its power. Prepared and privileged, staff said, the shelter regained power the next morning.

That same night, Helene ripped the roof off of Dixie County’s animal shelter, which is almost 100 miles away from North Central Florida’s Humane Society. HSNCF’s team drove over an hour to rescue 24 of its animals the day before the storm’s landfall.

The location, which usually shelters between 35 to 40 dogs, evacuated dogs from coastal shelters the days leading up to Helene. With just under a week’s notice of Helene, the shelter had to foster out all of its in-house animals to make room for more.

“Imagine if those dogs were still there during that time,” said HSNCF foster coordinator Valentina Landaeta. “That would have been very catastrophic.”

After working at HSNCF for almost six years, Landaeta said it has been through this process plenty of times. The week leading up to Helene’s landfall, HSNCF’s team traveled to places close to the coast that were expected to be hit beforehand, she said.

In addition to Dixie County, the shelter pulled animals from Levy County and transferred some from Madison County.

With a total pre-hurricane intake of 76 cats and dogs, HSNCF had to foster out all its existing animals beforehand. While the shelter typically struggles with finding dog fosters over cat fosters, it was able to find fosters for all in-house dogs, Landaeta said.

“Working here is just a mix of emotions all the time, [a] roller coaster,” she said. “But it's mostly just joy and happiness that we're able to provide a shelter for these animals.”

Before Helene hit, the shelter’s hurricane preparation began with an emergency plea.

Executive director Chelsea Bower said HSNCF sent an emergency plea for fosters, volunteers and supplies in a blast email the Tuesday before the storm. The shelter typically asks fosters to commit to an animal two to four weeks in advance based on the severity of a storm’s impact.

“This isn’t our first rodeo,” Bower said. “We’re really grateful that we have such a great community to help us be able to do things like pull 80 animals before a storm.”

Situated minutes away from the UF, HSNCF development director Franziska Raeber said part of the privilege comes with having students stepping up and helping out.

“It makes a huge difference for us,” Raeber said. “The community support is amazing, and the support we have with donations coming in, people donating off our Amazon list, fostering. It's amazing, and I couldn't be more grateful for this.”

Despite the overwhelming support, Raeber emphasized the continuous need for dog fosters.

While the shelter does not take in surrendered animals, it provides and pays for all supplies needed to foster one.

“If you are a student and you miss your pets at home, come and get a foster,” she said. “You are taking an animal out of a shelter and giving temporary relief, and in that moment, you're actually saving a life. Not the animal you're taking out, but you're creating a space for another animal to come in.”
"""

# Process the text using the transformer-based model
doc = nlp(article_text)

# 1. Extract Named Entities (no hardcoding, purely based on transformer model's output)
print("Named Entities and their types:")
for ent in doc.ents:
    print(f"{ent.label_} - {ent.text}")

# 2. Extract Coreference Chains (also purely based on the model's output)
print("\nCoreference Chains:")
if doc._.coref_chains:
    for chain in doc._.coref_chains:
        # Reconstruct the mention's text using token indices
        mentions = [" ".join([doc[i].text for i in mention]) for mention in chain.mentions]
        print(f"Coreference chain: {mentions}")
else:
    print("No coreference chains found.")

