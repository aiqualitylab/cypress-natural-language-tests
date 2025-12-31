// Requirement: Test shopping
// Test Type: Traditional

describe('Tests', function () {
    beforeEach(function () {
        cy.fixture('url_test_data').then((data) => {
            this.testData = data;
        });
    });

    it('should succeed with valid data', function () {
        cy.visit(this.testData.url);
        const valid = this.testData.test_cases.find(tc => tc.name === 'valid_test');
        const selectors = this.testData.selectors;
        
        Object.keys(selectors).forEach(field => {
            if (field !== 'submit' && valid[field]) {
                cy.get(selectors[field]).type(valid[field]);
            }
        });
        
        cy.get(selectors.submit).click();
    });

    it('should fail with invalid data', function () {
        cy.visit(this.testData.url);
        const invalid = this.testData.test_cases.find(tc => tc.name === 'invalid_test');
        const selectors = this.testData.selectors;
        
        Object.keys(selectors).forEach(field => {
            if (field !== 'submit' && invalid[field]) {
                cy.get(selectors[field]).type(invalid[field]);
            }
        });
        
        cy.get(selectors.submit).click();
    });
});