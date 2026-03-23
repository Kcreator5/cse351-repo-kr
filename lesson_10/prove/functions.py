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

# -----------------------------------------------------------------------------
def depth_fs_pedigree(family_id, tree):
    # KEEP this function even if you don't implement it
    # TODO - implement Depth first retrieval
    # Keep track of visited families to avoid cycles
    visited_families = set()
    
    def _dfs_recursive(fam_id):
        # Base case: if already visited or None, stop
        if fam_id is None or fam_id in visited_families:
            return
        
        visited_families.add(fam_id)
        
        # Request family data from server
        family_data = get_data_from_server(f'{TOP_API_URL}/family/{fam_id}')
        if family_data is None:
            return
        
        # Create Family object and add to tree
        family = Family(family_data)
        tree.add_family(family)
        print(f'Retrieved family: {fam_id}')
        
        # Request husband and wife data
        husband_data = get_data_from_server(f'{TOP_API_URL}/person/{family.get_husband()}')
        if husband_data:
            husband = Person(husband_data)
            tree.add_person(husband)
            print(f'Retrieved person: {husband.get_name()}')
            # Recurse on husband's parents
            _dfs_recursive(husband.get_parentid())
        
        wife_data = get_data_from_server(f'{TOP_API_URL}/person/{family.get_wife()}')
        if wife_data:
            wife = Person(wife_data)
            tree.add_person(wife)
            print(f'Retrieved person: {wife.get_name()}')
            # Recurse on wife's parents
            _dfs_recursive(wife.get_parentid())
        
        # Process children
        for child_id in family.get_children():
            child_data = get_data_from_server(f'{TOP_API_URL}/person/{child_id}')
            if child_data:
                child = Person(child_data)
                tree.add_person(child)
                print(f'Retrieved person: {child.get_name()}')
    
    # Start DFS from the initial family
    _dfs_recursive(family_id)

    # TODO - Printing out people and families that are retrieved from the server will help debugging

    pass

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