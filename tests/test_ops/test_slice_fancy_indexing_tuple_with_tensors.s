
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <r_5_5n1>:
	s_add_i32 s3, s15, -1                                      // 000000001600: 8103C10F
	s_load_b64 s[0:1], s[0:1], null                            // 000000001604: F4040000 F8000000
	s_cmp_gt_i32 s3, 2                                         // 00000000160C: BF028203
	s_mov_b32 s2, s15                                          // 000000001610: BE82000F
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001614: 986A80C1
	s_cmp_gt_i32 s15, 2                                        // 000000001618: BF02820F
	s_cselect_b32 s3, -1, 0                                    // 00000000161C: 980380C1
	s_delay_alu instid0(SALU_CYCLE_1) | instskip(SKIP_1) | instid1(SALU_CYCLE_1)// 000000001620: BF8704A9
	v_cndmask_b32_e64 v0, 0, 1, s3                             // 000000001624: D5010000 000D0280
	s_add_i32 s3, s15, 1                                       // 00000000162C: 8103810F
	s_cmp_gt_i32 s3, 2                                         // 000000001630: BF028203
	s_cselect_b32 s3, -1, 0                                    // 000000001634: 980380C1
	s_add_i32 s4, s15, 2                                       // 000000001638: 8104820F
	v_add_co_ci_u32_e32 v0, vcc_lo, 0, v0, vcc_lo              // 00000000163C: 40000080
	v_cndmask_b32_e64 v1, 0, 1, s3                             // 000000001640: D5010001 000D0280
	s_cmp_gt_i32 s4, 2                                         // 000000001648: BF028204
	s_cselect_b32 vcc_lo, -1, 0                                // 00000000164C: 986A80C1
	s_cmp_lt_u32 s15, 0x7ffffffd                               // 000000001650: BF0AFF0F 7FFFFFFD
	s_delay_alu instid0(VALU_DEP_1) | instskip(SKIP_3) | instid1(VALU_DEP_2)// 000000001658: BF870141
	v_add_co_ci_u32_e32 v0, vcc_lo, v0, v1, vcc_lo             // 00000000165C: 40000300
	s_cselect_b32 vcc_lo, -1, 0                                // 000000001660: 986A80C1
	s_ashr_i32 s3, s15, 31                                     // 000000001664: 86039F0F
	v_mov_b32_e32 v1, 0                                        // 000000001668: 7E020280
	v_add_co_ci_u32_e32 v0, vcc_lo, -1, v0, vcc_lo             // 00000000166C: 400000C1
	s_lshl_b64 s[2:3], s[2:3], 2                               // 000000001670: 84828202
	s_waitcnt lgkmcnt(0)                                       // 000000001674: BF89FC07
	s_add_u32 s0, s0, s2                                       // 000000001678: 80000200
	s_addc_u32 s1, s1, s3                                      // 00000000167C: 82010301
	global_store_b32 v1, v0, s[0:1]                            // 000000001680: DC6A0000 00000001
	s_nop 0                                                    // 000000001688: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 00000000168C: BFB60003
	s_endpgm                                                   // 000000001690: BFB00000
