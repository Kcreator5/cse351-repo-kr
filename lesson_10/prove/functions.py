"""
Course: CSE 351, week 10
File: functions.py
Author: Kevin

Instructions:

Depth First Search
https://www.youtube.com/watch?v=9RHO6jU--GU

Breadth First Search
https://www.youtube.com/watch?v=86g8jAQug04


Requesting a family from the server:
family_id = 6128784944
data = get_data_from_server('{TOP_API_URL}/family/{family_id}')

Example JSON returned from the server
{
    'id': 6128784944, 
    'husband_id': 2367673859,        # use with the Person API
    'wife_id': 2373686152,           # use with the Person API
    'children': [2380738417, 2185423094, 2192483455]    # use with the Person API
}

Requesting an individual from the server:
person_id = 2373686152
data = get_data_from_server('{TOP_API_URL}/person/{person_id}')

Example JSON returned from the server
{
    'id': 2373686152, 
    'name': 'Stella', 
    'birth': '9-3-1846', 
    'parent_id': 5428641880,   # use with the Family API
    'family_id': 6128784944    # use with the Family API
}


--------------------------------------------------------------------------------------
You will lose 10% if you don't detail your part 1 and part 2 code below

Describe how to speed up part 1

<Add your comments here>


Describe how to speed up part 2

<Add your comments here>


Extra (Optional) 10% Bonus to speed up part 3

<Add your comments here>

"""

from common import *
import queue
import threading

# -----------------------------------------------------------------------------
def depth_fs_pedigree(family_id, tree):

    visited_families = set()
    visited_people = set()

    lock = threading.Lock()

    def dfs(fam_id):
        if fam_id is None:
            return

        # Prevent duplicate work
        with lock:
            if fam_id in visited_families:
                return
            visited_families.add(fam_id)

        
        # 1. Get family from server
        
        family_data = get_data_from_server(f'{TOP_API_URL}/family/{fam_id}')
        if family_data is None:
            return

        family = Family(family_data)
        tree.add_family(family)

        threads = []


        # 2. Fetch people

        def fetch_person(person_id):
            if person_id is None:
                return None

            with lock:
                if person_id in visited_people:
                    return None
                visited_people.add(person_id)

            data = get_data_from_server(f'{TOP_API_URL}/person/{person_id}')
            if data is None:
                return None

            person = Person(data)
            tree.add_person(person)
            return person

        # Husband thread
        t1 = threading.Thread(target=fetch_person, args=(family.get_husband(),))
        threads.append(t1)
        t1.start()

        # Wife thread
        t2 = threading.Thread(target=fetch_person, args=(family.get_wife(),))
        threads.append(t2)
        t2.start()

        # Children threads
        for child_id in family.get_children():
            t = threading.Thread(target=fetch_person, args=(child_id,))
            threads.append(t)
            t.start()

        # Wait for all person fetches
        for t in threads:
            t.join()

        # 3. Recurse to parents

        # Need actual person objects from tree
        husband = tree.get_person(family.get_husband())
        wife = tree.get_person(family.get_wife())

        parent_threads = []

        if husband is not None:
            parent_threads.append(t)
            threading.Thread(target=dfs, args=(husband.get_parentid(),)).start()

        if wife is not None:
            parent_threads.append(t)
            threading.Thread(target=dfs, args=(wife.get_parentid(),)).start()

        # Wait for recursion threads
        for t in parent_threads:
            t.join()

    # Start DFS
    dfs(family_id)

# -----------------------------------------------------------------------------
def breadth_fs_pedigree(family_id, tree):
    # KEEP this function even if you don't implement it
    # TODO - implement breadth first retrieval
    # TODO - Printing out people and families that are retrieved from the server will help debugging

    pass

# -----------------------------------------------------------------------------
def breadth_fs_pedigree_limit5(family_id, tree):
    # KEEP this function even if you don't implement it
    # TODO - implement breadth first retrieval
    #      - Limit number of concurrent connections to the FS server to 5
    # TODO - Printing out people and families that are retrieved from the server will help debugging

    pass