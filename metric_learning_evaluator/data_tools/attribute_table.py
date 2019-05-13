"""Native Database used as AttributeTable
    The database implementation used for attribute container.

  There are two items stored in table
    - domain:   crossing reference condition
    - property: filtering condition
"""

import os
import sys
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

import json
import sqlite3
from glob import glob
from contextlib import closing

class AttributeTable(object):

    def __init__(self, db_path='attribute.db'):
        # init folders
        # init or check for base image extension
        # connect and init db (if necessary)
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS property
                (id            INTEGER  PRIMARY KEY  AUTOINCREMENT,
                 instance_id       INTEGER,
                 name          TEXT,
                 content       TEXT,
                 update_dt     TIMESTAMP  DEFAULT  CURRENT_TIMESTAMP)''')

            cur.execute('''
                CREATE TABLE IF NOT EXISTS domain
                (id            INTEGER  PRIMARY KEY  AUTOINCREMENT,
                 instance_id       INTEGER,
                 domain_tag    TEXT,
                 update_dt     TIMESTAMP  DEFAULT  CURRENT_TIMESTAMP)''')

            cur.execute('''
                    CREATE TRIGGER IF NOT EXISTS domain_update
                        AFTER UPDATE ON domain
                        FOR EACH ROW
                        WHEN NEW.update_dt <= OLD.update_dt
                    BEGIN
                        UPDATE domain SET update_dt=CURRENT_TIMESTAMP WHERE id=NEW.id;
                    END''')
            
            cur.execute('''
                    CREATE TRIGGER IF NOT EXISTS property_update
                        AFTER UPDATE ON property
                        FOR EACH ROW
                        WHEN NEW.update_dt <= OLD.update_dt                    BEGIN
                        UPDATE property SET update_dt=CURRENT_TIMESTAMP WHERE id=NEW.id;
                    END''')

    def insert_property(self, instance_id, name, content):
        assert isinstance(instance_id, int), 'type(instance_id) %s != int' % (type(instance_id))
        with closing(self.conn.cursor()) as cur:
            cur.execute('INSERT INTO property (instance_id, name, content) VALUES (?,?,?)', (instance_id, name, content,))

    def insert_domain(self, instance_id, domain_tag):
        assert isinstance(instance_id, int), 'type(instance_id) %s != int' % (type(instance_id))
        with closing(self.conn.cursor()) as cur:
            cur.execute('INSERT INTO domain (instance_id, domain_tag) VALUES (?, ?)', (instance_id, domain_tag,))

    def remove_property(self, instance_id, attr_name=None, attr_content=None):
        """ remove property with given instance_id """
        assert isinstance(instance_id, int), 'type(instance_id) %s != int' % (type(instance_id))
        with closing(self.conn.cursor()) as cur:
            if attr_name is not None and attr_content is not None:
                cur.execute('DELETE FROM property WHERE instance_id=? AND name=? AND content=?', (instance_id, attr_name, attr_content, ))
            else:
                cur.execute('DELETE FROM property WHERE instance_id = ?', (instance_id,))

    def remove_domain(self, instance_id, domain_tag=None):
        assert isinstance(instance_id, int), 'type(instance_id) %s != int' % (type(instance_id))
        with closing(self.conn.cursor()) as cur:
            if domain_tag is not None:
                cur.execute('DELETE FROM domain WHERE instance_id=? AND domain_tag=?', (instance_id, domain_tag,) )
            else:
                cur.execute('DELETE FROM domain WHERE instance_id=?', (instance_id,))

    def commit(self):
        self.conn.commit()

    def query_all_property_names(self):
        with closing(self.conn.cursor()) as cur:
            cur.execute('SELECT name FROM property')
            q = cur.fetchall()
        q = list(set(x[0] for x in q))
        return q

    def query_all_domain_names(self):
        with closing(self.conn.cursor()) as cur:
            cur.execute('SELECT domain_tag FROM domain')
            q = cur.fetchall()
        q = list(set(x[0] for x in q))
        return q

    def query_property_content_by_property_name(self, attr_names):
        """query property contents by attribute names """
        attr_names = self._query_input_converter(attr_names, str)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''SELECT DISTINCT content
                           FROM property
                           WHERE name in (%s)''' % (', '.join('?' for _ in attr_names)),
                        attr_names)
            q = cur.fetchall()
        q = [x[0] for x in q] # convert [(id,)] to [id, ]
        return q

    def query_instance_id_by_property_dict(self, attr_dict, domain_tag=None):
        """ query annotation ids by attribute name and contents (unicode)

            Args:
                attr_dict : a dictionary, keys are names, values are contents,
                            (e.g.) {'toppings':'sesame', 'category':'Pineapple_Bun'}
                domain_tag : str
            Returns:
                a list, contains anno id that meets "all" the conditions in attr_dict (logic : AND)
        """
        assert isinstance(attr_dict, dict), 'type(attr_dict) %s != dictionary' % (type(attr_dict))

        with closing(self.conn.cursor()) as cur:
            query_instance_ids = []
            if domain_tag:
                cur.execute('''SELECT instance_id
                            FROM domain
                            WHERE domain_tag=?
                            ''', (domain_tag,) )
                q = cur.fetchall()
                query_instance_ids.append([x[0] for x in q])

            for name,content in attr_dict.items():
                cur.execute('''SELECT instance_id
                            FROM property
                            WHERE name=? AND content=?
                            ''', (name, content,) )
                q = cur.fetchall()
                query_instance_ids.append([x[0] for x in q])

            if len(query_instance_ids) == 0:
                return []
            
            # get intersection of several query conditions
            query_instance_ids = list(set(query_instance_ids[0]).intersection(*query_instance_ids))
        return query_instance_ids


    def query_property_by_instance_ids(self, instance_ids):
        """query property by annotation ids"""
        instance_ids = self._query_input_converter(instance_ids, int)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''SELECT name, content
                           FROM property
                           WHERE instance_id in (%s)
                           ''' % (', '.join(str(e) for e in instance_ids)))
                        
            q = cur.fetchall()
        return [{name:content} for (name, content) in q] 

    def query_domain_by_instance_ids(self, instance_ids):
        """query domain by annotation ids"""
        instance_ids = self._query_input_converter(instance_ids, int)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''SELECT domain_tag
                           FROM domain
                           WHERE instance_id in (%s)
                           ''' % (', '.join(str(e) for e in instance_ids)))
                        
            q = cur.fetchall()
        return [x[0] for x in q]

    def _query_input_converter(self, queries, element_type=None):
        """unify input in the forms of 'single value', 'numpy array', 'list' to 'list'

        for example, 210 will be converted to [210], np.array[3, 5, 2] will be converted to [3, 5, 2]
        while ['a', 'k', 'e'] will be kept the same (already a list)

        Args:
            queries: input (can take in the form of 'single value', 'np.ndarry', 'list')
            element_type: element type that will be checked (e.g. int, str, unicode)

        Returns:
            queries: list of input elements [element_1, element_2, ...]
        """
        if isinstance(queries, element_type):
            queries = [queries]
        elif isinstance(queries, list):
            pass
        elif isinstance(queries, np.ndarray):
            queries = queries.flatten().tolist()
        else:
            assert False, 'input_query only support %s, list, or ndarray != %s' % (element_type, type(queries))
        for i, query in enumerate(queries):
            assert isinstance(query, element_type), 'type(input[%d]) %s != %s' % (type(query), element_type)
        return queries


if __name__ == "__main__":
    
    # define data
    instance_idA = 10
    attr_A = {'group' : '1', 'toppings':'sesame'}
    domainA = 'seen'

    instance_idB = 12
    attr_B = {'group' : '2', 'toppings':'sesame'}
    domainB = 'unseen'

    # add into db
    db = AttributeTable('Attr.db')

    db.insert_domain(instance_idA, domainA)
    for name, content in attr_A.items():
        db.insert_property(instance_idA, name, content)
    
    db.insert_domain(instance_idB, domainB)
    for name, content in attr_B.items():
        db.insert_property(instance_idB, name, content)

    # db.commit()

    # test query
    attr_names = db.query_all_property_names()
    attr_contents = db.query_property_content_by_property_name(attr_names)
    print("Attribute names", attr_names)        
    print("Attribute contents", attr_contents)      


    instance_ids = db.query_instance_id_by_property_dict({'toppings':'sesame'} , 'seen')       
    print(instance_ids)

    instance_ids = db.query_instance_id_by_property_dict({'toppings':'sesame', 'group': '2'} , 'unseen')       
    print(instance_ids)

    instance_ids = db.query_instance_id_by_property_dict({'toppings':'sesame'} )       
    print(instance_ids)

    instance_ids = db.query_instance_id_by_property_dict({'group':'1'} )       
    print(instance_ids)

    attr = db.query_property_by_instance_ids([10,12])
    print(attr)
   
    domain = db.query_domain_by_instance_id(instance_idA)
    print(domain)

    # db.remove_domain(10, 'seen')
    #db.remove_property(10, 'group', '100')
    db.remove_property(10)
    print(db.query_property_by_instance_ids(10))
    domain = db.query_domain_by_instance_id(10)
    print(domain)