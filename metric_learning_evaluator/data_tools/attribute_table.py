import os
import json
import sqlite3
from glob import glob
from contextlib import closing

class AttributeTable(object):
    """
      Light-weighted table
    """

    def __init__(self, db_path='attribute.db'):
        # init folders
        # init or check for base image extension
        # connect and init db (if necessary)
        self.conn = sqlite3.connect(db_path, timeout=30.0, check_same_thread=False)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''
                CREATE TABLE IF NOT EXISTS attributes
                (id            INTEGER  PRIMARY KEY  AUTOINCREMENT,
                 instance_id       INTEGER  REFERENCES  annotations(id),
                 name          TEXT,
                 content       TEXT,
                 update_dt     TIMESTAMP  DEFAULT  CURRENT_TIMESTAMP)''')
        
            cur.execute('''
                    CREATE TRIGGER IF NOT EXISTS attributes_update
                        AFTER UPDATE ON attributes
                        FOR EACH ROW
                        WHEN NEW.update_dt <= OLD.update_dt
                    BEGIN
                        UPDATE attributes SET update_dt=CURRENT_TIMESTAMP WHERE id=NEW.id;
                    END''')
    
    def update_attribute(self, instance_id, attr_name, attr_content):
        """ update an annotation's attribute """
        assert isinstance(instance_id, int), 'type(instance_id) %s != int' % (type(instance_id))
        with closing(self.conn.cursor()) as cur:
            cur.execute('INSERT INTO attributes(instance_id, name, content) VALUES (?,?,?)', (instance_id, attr_name, attr_content,))


    def query_all_attribute_names(self):
        with closing(self.conn.cursor()) as cur:
            cur.execute('SELECT name FROM attributes')
            q = cur.fetchall()
        q = list(set(x[0] for x in q))
        return q 

    def query_attr_content_by_attr_name(self, attr_names):
        """query attribute contents by attribute names """
        attr_names = self._query_input_converter(attr_names, str)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''SELECT DISTINCT attributes.content
                           FROM attributes
                           WHERE attributes.name in (%s)''' % (', '.join('?' for _ in attr_names)),
                        attr_names)
            q = cur.fetchall()
        q = [x[0] for x in q] # convert [(id,)] to [id, ]
        return q


    def query_instance_id_by_attribute_dict(self, attr_dict):
        """ query annotation ids by attribute name and contents (unicode)

            Args:
                attr_dict : a dictionary, keys are names, values are contents,
                            (e.g.) {'toppings':'sesame', 'category':'Pineapple_Bun'}
            Returns:
                a list, contains anno id that meets "all" the conditions in attr_dict (logic : AND)
        """
        assert isinstance(attr_dict, dict), 'type(attr_dict) %s != dictionary' % (type(attr_dict))

        with closing(self.conn.cursor()) as cur:
            query_instance_ids = []
            
            for name,content in attr_dict.items():
              
                cur.execute('''SELECT attributes.instance_id
                            FROM attributes
                            WHERE attributes.name = '%s' AND attributes.content = '%s' 
                            ''' % (name, content) )
                q = cur.fetchall()
                query_instance_ids.append([x[0] for x in q])

            if len(query_instance_ids) == 0:
                return []
            
            # get intersection of several query conditions
            query_instance_ids = list(set(query_instance_ids[0]).intersection(*query_instance_ids))
        return query_instance_ids


    def query_attr_by_instance_id(self, instance_ids):
        """query annotation information by annotation ids"""
        instance_ids = self._query_input_converter(instance_ids, int)
        with closing(self.conn.cursor()) as cur:
            cur.execute('''SELECT GROUP_CONCAT(attributes.name,','), 
                                  GROUP_CONCAT(attributes.content,',')
                           FROM attributes
                           WHERE attributes.instance_id in (%s)
                           GROUP BY attributes.instance_id''' % (', '.join(str(e) for e in instance_ids)))
                        
            q = cur.fetchall()
        
        q = [ { name:content for (name,content) in 
            zip(x[0].split(','),x[1].split(',')) } if x[1] is not None else {} 
            for x in q]  # convert tuple to dict
        return q

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
    instance_idA = 10
    attr_A = {'group' : '1', 'toppings':'sesame'}
    instance_idB = 12
    attr_B = {'group' : '2', 'toppings':'sesame'}


    db = AttributeTable()

    for name, content in attr_A.items():
        db.update_attribute(instance_idA, name, content)
    for name, content in attr_B.items():
        db.update_attribute(instance_idB, name, content)

    attr_names = db.query_all_attribute_names()
    attr_contents = db.query_attr_content_by_attr_name(attr_names)
    print("Attribute names", attr_names)        
    print("Attribute contents", attr_contents)      


    instance_ids = db.query_instance_id_by_attribute_dict({'toppings':'sesame'} )       
    print(instance_ids)

    attr = db.query_attr_by_instance_id([10,12])
    print(attr)
   