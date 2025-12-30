import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# 1. Mock 'aqt' and 'aqt.qt' BEFORE importing main
# We need to mock broadly because main.py imports classes from aqt.qt directly

class MockQtModule:
    # Logic for mocking Qt classes used in main.py
    # key classes: QDialog, QLineEdit, etc.
    class QDialog:
        class DialogCode:
            Accepted = 1
            Rejected = 0
            
        def __init__(self, parent=None): pass
        def setWindowTitle(self, t): pass
        def setMinimumWidth(self, w): pass
        def setLayout(self, l): pass
        def setModal(self, b): pass
        def show(self): pass
        def close(self): pass
        def exec(self): return 1 # Accepted
        def reject(self): pass
        def accept(self): pass

    class QLineEdit:
        def __init__(self, text=""): self._text = text
        def text(self): return self._text
        def setText(self, t): self._text = t

    class QSpinBox:
        def setRange(self, a, b): pass
        def setValue(self, v): self._val = v
        def value(self): return self._val 

    class QLabel:
        def __init__(self, t=""): pass
        def setAlignment(self, a): pass
        def setText(self, t): pass

    class QPushButton:
        def __init__(self, t=""): 
            self.clicked = MagicMock()
        def setEnabled(self, b): pass

    # Layouts
    class QVBoxLayout:
        def addLayout(self, l): pass
        def addWidget(self, w): pass
    class QHBoxLayout:
        def addWidget(self, w): pass
    class QFormLayout:
        def addRow(self, l, w): pass
        
    class Qt:
        class AlignmentFlag:
            AlignCenter = 0

    class QProgressBar:
        def setRange(self, a, b): pass
        def setValue(self, v): pass

    class QApplication:
        @staticmethod
        def processEvents(): pass
        
    class QAction:
        def __init__(self, t, p): 
            self.triggered = MagicMock()
            
    class QInputDialog:
        @staticmethod
        def getItem(p, t, l, items, c, e):
            return (items[0], True) if items else (None, False)

    class QEventLoop: pass
    class QTimer: pass
    
    # Removed erroneous lines


# Apply mocks
sys.modules['aqt'] = MagicMock()
sys.modules['aqt.qt'] = MockQtModule()
sys.modules['aqt.utils'] = MagicMock()
sys.modules['aqt.gui_hooks'] = MagicMock()

# Now we can safely import main
# We must ensure config/graph are importable.
# Assuming test_anki_mock.py is in the addon root.
import main

class TestAnkiUI(unittest.TestCase):
    
    @patch('main.load_news_articles')
    def test_news_filter_dialog_fetch(self, mock_load):
        # Setup mock return
        mock_load.return_value = [{"title": "Test Article", "source": {"name": "Test"}}]
        
        # Instantiate Dialog
        dialog = main.NewsFilterDialog()
        
        # Check defaults
        self.assertEqual(dialog.query_input.text(), "technology")
        self.assertEqual(dialog.days_input.value(), 7)
        
        # Trigger fetch
        dialog.on_fetch()
        
        # Verify load_news_articles called with correct params
        mock_load.assert_called_with(query="technology", days_back=7, limit=50)
        
        # Verify articles stored
        self.assertEqual(len(dialog.articles), 1)
        self.assertEqual(dialog.articles[0]["title"], "Test Article")

    @patch('main.create_quiz_graph')
    @patch('main.NewsFilterDialog')
    @patch('main.mw') # Anki main window mock
    def test_generate_mcq_flow(self, mock_mw, MockDialogClass, mock_create_graph):
        # 1. Setup Mock Dialog to return articles
        mock_dialog_instance = MockDialogClass.return_value
        mock_dialog_instance.exec.return_value = 1 # Accepted
        mock_dialog_instance.get_articles.return_value = [
            {"title": "Art1", "description": "Desc1"}
        ]
        
        # 2. Setup Mock Graph
        mock_app = MagicMock()
        mock_create_graph.return_value = mock_app
        mock_app.invoke.return_value = {
            "sentence": "Gen Sentence",
            "sentence_masked": "Gen Masked",
            "synonyms": ["syn1", "syn2"],
            "explanation": "Exp",
            "word_length": 5, 
            "first_letter": "G"
        }
        
        # 3. Setup Mock Card/Note
        mock_note = MagicMock()
        # Make it behave like a dict
        _note_data = {
            "Word": "TestWord", 
            "Back": "Def",
            "SentenceBlank": "", "OptionA": "", "Answer": ""
        }
        mock_note.__getitem__.side_effect = _note_data.__getitem__
        mock_note.__setitem__.side_effect = _note_data.__setitem__
        mock_note.keys.side_effect = _note_data.keys
        mock_note.get.side_effect = _note_data.get
        mock_note.flush = MagicMock()
        
        # Card.note() returns the dict-like note
        mock_card = MagicMock()
        mock_card.note.return_value = mock_note
        
        # mw.col.getCard(cid)
        mock_mw.col.getCard.return_value = mock_card
        
        # 4. Run Function
        cids = [123]
        main.generate_mcq_for_cards(cids)
        
        # 5. Assertions
        # Dialog shown?
        mock_dialog_instance.exec.assert_called_once()
        
        # Graph compiled?
        mock_create_graph.assert_called_once()
        
        # Graph invoked?
        # Check the state passed to invoke
        args, _ = mock_app.invoke.call_args
        initial_state = args[0]
        self.assertEqual(initial_state["word"], "TestWord")
        self.assertEqual(initial_state["article"]["title"], "Art1")
        
        # Note updated?
        self.assertEqual(mock_note["Answer"], "Gen Sentence")
        self.assertEqual(mock_note["OptionA"], "syn1")
        # Ensure mapping correct
        self.assertEqual(mock_note["SentenceBlank"], "Gen Masked")
        
        print("\n[SUCCESS] TestAnkiUI: mock flow verified.")


if __name__ == '__main__':
    unittest.main()
