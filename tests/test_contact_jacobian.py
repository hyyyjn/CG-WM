import unittest

import torch

from stage2.differentiable_complementarity_free_contact_dynamics import (
    contact_delassus_matrix,
    contact_jacobian_rows,
    regularized_contact_solve,
    rigid_inverse_mass_matrix,
)


class ContactJacobianTests(unittest.TestCase):
    def test_regularized_contact_solve_handles_singular_redundant_facets(self):
        matrix = torch.ones((3, 3), requires_grad=True)
        solution = regularized_contact_solve(matrix, torch.ones((3, 1)))
        self.assertTrue(bool(torch.isfinite(solution).all()))
        solution.sum().backward()
        self.assertTrue(bool(torch.isfinite(matrix.grad).all()))

    def test_off_center_contact_contains_angular_jacobian(self):
        direction = torch.tensor([[0.0, 0.0, 1.0]])
        lever = torch.tensor([[1.0, 0.0, 0.0]])
        jacobian = contact_jacobian_rows(direction, lever)
        self.assertTrue(torch.allclose(
            jacobian,
            torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0, 0.0]]),
        ))

    def test_delassus_includes_linear_and_rotational_effective_mass(self):
        jacobian = contact_jacobian_rows(
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.tensor([[1.0, 0.0, 0.0]]),
        )
        inverse_mass = rigid_inverse_mass_matrix(
            torch.tensor(2.0), torch.tensor([4.0, 4.0, 4.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), dynamic=True,
        )
        delassus = contact_delassus_matrix(jacobian, inverse_mass)
        self.assertAlmostEqual(float(delassus[0, 0]), 0.75, places=6)
        self.assertAlmostEqual(float(1.0 / delassus[0, 0]), 4.0 / 3.0, places=6)

    def test_static_body_has_zero_inverse_generalized_mass(self):
        inverse_mass = rigid_inverse_mass_matrix(
            torch.tensor(1.0), torch.ones(3),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), dynamic=False,
        )
        self.assertEqual(int(torch.count_nonzero(inverse_mass)), 0)

    def test_contact_space_matrix_remains_differentiable(self):
        mass = torch.tensor(2.0, requires_grad=True)
        inverse_mass = rigid_inverse_mass_matrix(
            mass, torch.tensor([4.0, 4.0, 4.0]),
            torch.tensor([1.0, 0.0, 0.0, 0.0]), dynamic=True,
        )
        jacobian = contact_jacobian_rows(torch.tensor([[0.0, 0.0, 1.0]]), torch.zeros(1, 3))
        contact_delassus_matrix(jacobian, inverse_mass).sum().backward()
        self.assertIsNotNone(mass.grad)
        self.assertLess(float(mass.grad), 0.0)


if __name__ == "__main__":
    unittest.main()
