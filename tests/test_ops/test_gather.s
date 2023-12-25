
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_4_4>:
	s_add_i32 s3, s15, -1                                      // 000000001600: 8103C10F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_cmp_gt_i32 s3, 1                                         // 00000000160C: BF028103
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_cselect_b32 s3, -1, 0                                    // 000000001614: 980380C1
	s_cmp_gt_i32 s15, 1                                        // 000000001618: BF02810F
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 00000000161C: D5010000 000D0280
	s_cselect_b32 s3, -1, 0                                    // 000000001624: 980380C1
	s_add_i32 s4, s15, 1                                       // 000000001628: 8104810F
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 00000000162C: D5010001 000D0280
	s_cmp_gt_i32 s4, 1                                         // 000000001634: BF028104
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001638: 986A80C1
	s_cmp_lt_u32 s15, 0x7ffffffe                               // 00000000163C: BF0AFF0F 7FFFFFFE
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001644: BF870141
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 000000001648: 40000300
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000164C: 986A80C1
	s_ashr_i32 s3, s15, 31                                     // 000000001650: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 000000001654: 7E020280
	v_add_co_ci_u32_e32 v0, vcc_lo, -1, v0, vcc_lo             // 000000001658: 400000C1
	s_lshl_b64 s[2:3], s[2:3], 2                               // 00000000165C: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001660: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001664: 80000200
	s_addc_u32 s1, s1, s3                                      // 000000001668: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 00000000166C: DC6A0000 00000001
	s_nop 0                                                    // 000000001674: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 000000001678: BFB60003
	s_endpgm                                                   // 00000000167C: BFB00000
