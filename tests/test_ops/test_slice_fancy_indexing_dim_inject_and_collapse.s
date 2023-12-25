
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_6_6>:
	s_add_i32 s3, s15, -1                                      // 000000001600: 8103C10F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_cmp_gt_i32 s3, 3                                         // 00000000160C: BF028303
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_cselect_b32 s3, -1, 0                                    // 000000001614: 980380C1
	s_cmp_gt_i32 s15, 3                                        // 000000001618: BF02830F
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 00000000161C: D5010000 000D0280
	s_cselect_b32 s3, -1, 0                                    // 000000001624: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001628: BF8704A9
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 00000000162C: D5010001 000D0280
	s_add_i32 s3, s15, 1                                       // 000000001634: 8103810F
	s_cmp_gt_i32 s3, 3                                         // 000000001638: BF028303
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000163C: 986A80C1
	s_add_i32 s3, s15, 2                                       // 000000001640: 8103820F
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001644: 40000300
	s_cmp_gt_i32 s3, 3                                         // 000000001648: BF028303
	s_cselect_b32 s3, -1, 0                                    // 00000000164C: 980380C1
	s_add_i32 s4, s15, 3                                       // 000000001650: 8104830F
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 000000001654: D5010001 000D0280
	s_cmp_gt_i32 s4, 3                                         // 00000000165C: BF028304
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001660: 986A80C1
	s_cmp_lt_u32 s15, 0x7ffffffc                               // 000000001664: BF0AFF0F 7FFFFFFC
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 00000000166C: BF870141
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001670: 40000300
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001674: 986A80C1
	s_ashr_i32 s3, s15, 31                                     // 000000001678: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 00000000167C: 7E020280
	v_add_co_ci_u32_e32 v0, vcc_lo, -1, v0, vcc_lo             // 000000001680: 400000C1
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001684: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001688: BF89FC07
	s_add_u32 s0, s0, s2                                       // 00000000168C: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001690: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001694: DC6A0000 00000001
	s_nop 0                                                    // 00000000169C: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016A0: BFB60003
	s_endpgm                                                   // 0000000016A4: BFB00000
