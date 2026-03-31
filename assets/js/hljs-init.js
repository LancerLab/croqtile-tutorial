document.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('pre code[class*="language-"]').forEach(function(el) {
    hljs.highlightElement(el);
  });
});
