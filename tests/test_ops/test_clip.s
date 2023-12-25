
<stdin>:	file format elf64-amdgpu

Disassembly of section .text:

0000000000001600 <E_2925n4>:
	s_load_b128 s[0:3], s[0:1], null                           // 000000001600: F4080000 F8000000
	s_mov_b32 s4, s15                                          // 000000001608: BE84000F
	s_ashr_i32 s5, s15, 31                                     // 00000000160C: 86059F0F
	s_delay_alu instid0(SALU_CYCLE_1)                          // 000000001610: BF870009
	s_lshl_b64 s[4:5], s[4:5], 2                               // 000000001614: 84848204
	s_waitcnt lgkmcnt(0)                                       // 000000001618: BF89FC07
	s_add_u32 s2, s2, s4                                       // 00000000161C: 80020402
	s_addc_u32 s3, s3, s5                                      // 000000001620: 82030503
	s_add_u32 s0, s0, s4                                       // 000000001624: 80000400
	s_load_b32 s2, s[2:3], null                                // 000000001628: F4000081 F8000000
	s_addc_u32 s1, s1, s5                                      // 000000001630: 82010501
	s_waitcnt lgkmcnt(0)                                       // 000000001634: BF89FC07
	v_add_f32_e64 v0, 0xc0133333, s2                           // 000000001638: D5030000 000004FF C0133333
	v_cmp_lt_f32_e64 s3, 0xc0133333, s2                        // 000000001644: D4110003 000004FF C0133333
	v_cmp_ngt_f32_e64 vcc_lo, 0xc0133333, s2                   // 000000001650: D41B006A 000004FF C0133333
	s_delay_alu instid0(VALU_DEP_3) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000165C: BF870093
	v_mul_f32_e32 v0, 0.5, v0                                  // 000000001660: 100000F0
	v_cndmask_b32_e64 v0, v0, s2, s3                           // 000000001664: D5010000 000C0500
	s_delay_alu instid0(VALU_DEP_1) | instskip(NEXT) | instid1(VALU_DEP_1)// 00000000166C: BF870091
	v_cndmask_b32_e32 v0, 0xc0133333, v0, vcc_lo               // 000000001670: 020000FF C0133333
	v_sub_f32_e32 v1, 0xbf99999a, v0                           // 000000001678: 080200FF BF99999A
	v_cmp_gt_f32_e32 vcc_lo, 0x3f99999a, v0                    // 000000001680: 7C2800FF 3F99999A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 000000001688: BF870092
	v_mul_f32_e32 v1, 0.5, v1                                  // 00000000168C: 100202F0
	v_cndmask_b32_e64 v1, v1, -v0, vcc_lo                      // 000000001690: D5010001 41AA0101
	v_cmp_nlt_f32_e32 vcc_lo, 0x3f99999a, v0                   // 000000001698: 7C3C00FF 3F99999A
	s_delay_alu instid0(VALU_DEP_2) | instskip(NEXT) | instid1(VALU_DEP_1)// 0000000016A0: BF870092
	v_dual_cndmask_b32 v0, 0xbf99999a, v1 :: v_dual_mov_b32 v1, 0// 0000000016A4: CA5002FF 00000080 BF99999A
	v_xor_b32_e32 v0, 0x80000000, v0                           // 0000000016B0: 3A0000FF 80000000
	global_store_b32 v1, v0, s[0:1]                            // 0000000016B8: DC6A0000 00000001
	s_nop 0                                                    // 0000000016C0: BF800000
	s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)                       // 0000000016C4: BFB60003
	s_endpgm                                                   // 0000000016C8: BFB00000
