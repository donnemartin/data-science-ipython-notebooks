from nose.tools import assert_equal
from ..transform_util import TransformUtil


class TestTransformUtil():

    states = [' Alabama ', 'Georgia!', 'Georgia', 'georgia', \
          'FlOrIda', 'south carolina##', 'West virginia?']
    
    expected_output = ['Alabama',
                       'Georgia',
                       'Georgia',
                       'Georgia',
                       'Florida',
                       'South Carolina',
                       'West Virginia']
    
    def test_remove_punctuation(self):
        assert_equal(TransformUtil.remove_punctuation('!#?'), '')
        
    def test_map_remove_punctuation(self):
        # Map applies a function to a collection
        output = map(TransformUtil.remove_punctuation, self.states)
        assert_equal('!#?' not in output, True)

    def test_clean_strings(self):
        clean_ops = [str.strip, TransformUtil.remove_punctuation, str.title] 
        output = TransformUtil.clean_strings(self.states, clean_ops)
        assert_equal(output, self.expected_output)